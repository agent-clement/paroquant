from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from paroquant.optim.util import get_named_linears, set_module_by_name

_AWQ_REORDER = (0, 2, 4, 6, 1, 3, 5, 7)
_LAYER_PATHS = ["model.layers", "model.language_model.layers", "language_model.layers"]
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _parse_dtype_arg(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    return _DTYPES[dtype]


def _dtype_name(dtype: torch.dtype) -> str:
    for name, value in _DTYPES.items():
        if value == dtype:
            return name
    raise ValueError(f"Unsupported dtype: {dtype}")


def _get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    for tensor in model.parameters():
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in model.buffers():
        if tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def _canonicalize_tensor_key(key: str) -> str:
    if key.startswith("model.language_model.visual."):
        key = "model.visual." + key.removeprefix("model.language_model.visual.")
    elif key.startswith("model.language_model.language_model.language_model."):
        key = "model.language_model." + key.removeprefix("model.language_model.language_model.language_model.")

    parts = key.split(".")
    canonical = []
    for part in parts:
        if part == "language_model" and canonical and canonical[-1] == "language_model":
            continue
        canonical.append(part)
    return ".".join(canonical)


def _canonicalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    canonical: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        canonical_key = _canonicalize_tensor_key(key)
        if canonical_key in canonical and canonical_key != key:
            raise ValueError(f"State dict key collision after canonicalization: {key} -> {canonical_key}")
        canonical[canonical_key] = value
    return canonical


def _load_model(model_id: str, device_map: str = "cpu", dtype: torch.dtype | str = "auto") -> torch.nn.Module:
    kwargs = dict(dtype=dtype, device_map=device_map, low_cpu_mem_usage=True, trust_remote_code=True)
    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def _save_preprocessors(model_id: str, output_path: Path) -> None:
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor.save_pretrained(output_path)
        if getattr(processor, "image_processor", None) is not None:
            processor.image_processor.save_pretrained(output_path)
        if getattr(processor, "video_processor", None) is not None:
            processor.video_processor.save_pretrained(output_path)
    except Exception:
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(output_path)


def _get_blocks(model: torch.nn.Module):
    for path in _LAYER_PATHS:
        node = model
        try:
            for attr in path.split("."):
                node = getattr(node, attr)
            return node
        except AttributeError:
            continue
    raise NotImplementedError(f"Unsupported model structure: {type(model)}")


def _get_value(state_dict: dict, *keys: str):
    for key in keys:
        if key in state_dict:
            val = state_dict[key]
            return int(val.item()) if isinstance(val, torch.Tensor) else val
    raise KeyError(f"None of {keys} found")


def _stack_if_numbered(state_dict: dict, key: str) -> torch.Tensor:
    if key in state_dict:
        return state_dict[key]
    parts = []
    i = 0
    while f"{key}.{i}" in state_dict:
        parts.append(state_dict[f"{key}.{i}"])
        i += 1
    if parts:
        return torch.stack(parts)
    raise KeyError(key)


def _pack_awq(values: torch.Tensor, bits: int = 4) -> torch.Tensor:
    pack_factor = 32 // bits
    reordered = values.to(torch.int32).view(values.shape[0], -1, pack_factor)[:, :, _AWQ_REORDER]
    packed = torch.zeros(reordered.shape[0], reordered.shape[1], dtype=torch.int32, device=values.device)
    for i in range(pack_factor):
        packed |= (reordered[:, :, i] & 0xF) << (bits * i)
    return packed


@torch.no_grad()
def _convert_pseudo(model: torch.nn.Module, result_dir: Path) -> int:
    from paroquant.optim.qlinear import PseudoQuantizedLinear

    blocks = _get_blocks(model)
    count = 0
    for layer_idx, layer in enumerate(tqdm(blocks, desc="Pseudo-quantizing")):
        layer = layer.cuda()
        for name, module in get_named_linears(layer).items():
            pt_file = result_dir / f"{layer_idx}.{name}.pt"
            if not pt_file.exists():
                continue
            sd = torch.load(pt_file, weights_only=False, map_location="cuda")
            module.weight.data.copy_(PseudoQuantizedLinear.from_state_dict(sd).pseudo_weight())
            count += 1
        layer.cpu()
    return count


@torch.no_grad()
def _quantize_layer(state_dict: dict, device: str, buffer_dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], int, int, int]:
    from paroquant.kernels.cuda import scaled_pairwise_rotation

    weight = state_dict["weight"].to(device=device, dtype=torch.float32)
    out_features, in_features = weight.shape

    bits = int(_get_value(state_dict, "n_bits", "quantizer.n_bits"))
    group_size = int(_get_value(state_dict, "group_size", "quantizer.group_size"))

    pairs = _stack_if_numbered(state_dict, "pairs_grouped").to(device=device, dtype=torch.short)
    theta = _stack_if_numbered(state_dict, "angles_grouped").to(device=device, dtype=torch.float32)

    channel_scales_opt = state_dict["channel_scales"].to(device=device, dtype=torch.float32)
    if channel_scales_opt.ndim == 1:
        channel_scales_opt = channel_scales_opt.unsqueeze(0)

    rotated = scaled_pairwise_rotation(weight * channel_scales_opt, pairs, theta, None, group_size)

    n_groups = in_features // group_size
    scales_flat = state_dict["quantizer.scale"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    zp_flat = state_dict["quantizer.zero_point_float"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    zero_points = torch.clamp(-torch.round(zp_flat), 0, (1 << bits) - 1)

    quantized = (
        torch.clamp(
            torch.round(rotated.reshape(-1, group_size) / scales_flat) + zero_points,
            0,
            (1 << bits) - 1,
        )
        .to(torch.int32)
        .reshape(out_features, in_features)
    )

    channel_scales = (1.0 / channel_scales_opt).to(buffer_dtype).cpu()
    if channel_scales.ndim == 1:
        channel_scales = channel_scales.unsqueeze(0)

    buffers: dict[str, torch.Tensor] = {
        "qweight": _pack_awq(quantized.T.contiguous()).cpu(),
        "qzeros": _pack_awq(zero_points.to(torch.int32).reshape(out_features, n_groups).T.contiguous()).cpu(),
        "scales": scales_flat.reshape(out_features, n_groups).T.contiguous().to(buffer_dtype).cpu(),
        "theta": theta.to(buffer_dtype).cpu(),
        "pairs": pairs.cpu(),
        "channel_scales": channel_scales,
    }
    if "bias" in state_dict and state_dict["bias"] is not None:
        buffers["bias"] = state_dict["bias"].to(buffer_dtype).cpu()

    return buffers, bits, group_size, int(theta.shape[0])


@torch.no_grad()
def _convert_real(model: torch.nn.Module, result_dir: Path, *, buffer_dtype: torch.dtype) -> tuple[int, dict[str, Any]]:
    from paroquant.inference.backends.transformers.modules import RotateQuantizedLinear

    blocks = _get_blocks(model)
    count = 0
    bits = group_size = krot = 0

    for layer_idx, layer in enumerate(tqdm(blocks, desc="Quantizing")):
        for name, module in get_named_linears(layer).items():
            pt_file = result_dir / f"{layer_idx}.{name}.pt"
            if not pt_file.exists():
                continue

            sd = torch.load(pt_file, map_location="cpu", weights_only=False)
            buffers, bits, group_size, krot = _quantize_layer(sd, device="cuda", buffer_dtype=buffer_dtype)

            rl = RotateQuantizedLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                group_size=group_size,
                bits=bits,
                krot=krot,
            )
            rl = rl.to(dtype=buffer_dtype)
            rl.load_state_dict(buffers, strict=False)
            set_module_by_name(layer, name, rl)
            count += 1

        layer.cpu()
        torch.cuda.empty_cache()

    quant_config: dict[str, Any] = {
        "quant_method": "paroquant",
        "bits": bits,
        "group_size": group_size,
        "krot": krot,
        "storage_dtype": _dtype_name(buffer_dtype),
    }
    return count, quant_config


@torch.no_grad()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--mode", choices=["real", "pseudo"], default="real")
    parser.add_argument("--dtype", choices=["auto", *_DTYPES], default="auto")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    load_dtype = _parse_dtype_arg(args.dtype)
    model = _load_model(args.model, device_map="cpu" if args.mode == "real" else "cuda", dtype=load_dtype)
    export_dtype = _get_model_dtype(model) if load_dtype == "auto" else load_dtype

    if args.mode == "pseudo":
        count = _convert_pseudo(model, result_dir)
    else:
        count, quant_config = _convert_real(model, result_dir, buffer_dtype=export_dtype)
        args_json = result_dir / "args.json"
        if args_json.exists():
            skipped = json.loads(args_json.read_text()).get("skipped_modules", [])
            if skipped:
                quant_config["modules_to_not_convert"] = skipped
        model.config.quantization_config = quant_config

    if count == 0:
        raise RuntimeError(f"No checkpoint files matched in {result_dir}")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    state_dict = _canonicalize_state_dict_keys(model.state_dict())
    model.save_pretrained(output_path, state_dict=state_dict, save_original_format=False)
    _save_preprocessors(args.model, output_path)

    print(f"Converted {count} layers ({args.mode}) → {output_path}")


if __name__ == "__main__":
    main()
