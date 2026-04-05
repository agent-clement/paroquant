from __future__ import annotations

import gc
import logging
import math
import random
from dataclasses import dataclass
import warnings
from typing import TypeVar
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class CalibDatasetSpec:
    kind: str
    dataset_id: str
    config_name: str | None = None
    split: str | None = None
    text_key: str = "text"
    text_keys: tuple[str, ...] = ("text",)
    weight: float = 1.0
    revision: str | None = None
    trust_remote_code: bool = False
    repeat: bool = False


def _last_query_value(query: dict[str, list[str]], *keys: str) -> str | None:
    for key in keys:
        if key in query and query[key]:
            return query[key][-1]
    return None


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def parse_calib_dataset_spec(spec: str, *, default_split: str) -> CalibDatasetSpec:
    parsed = urlparse(spec)
    query = parse_qs(parsed.query, keep_blank_values=False)

    raw_weight = _last_query_value(query, "weight")
    if raw_weight is None:
        weight = 1.0
    else:
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid calibration dataset weight {raw_weight!r} in {spec!r}") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(
            f"Calibration dataset weight must be a finite positive number, got {raw_weight!r} for {spec!r}"
        )

    split = _last_query_value(query, "split") or default_split
    text_key = _last_query_value(query, "text_key", "text_column", "field") or "text"
    text_keys = tuple(key.strip() for key in text_key.split(",") if key.strip())
    if len(text_keys) == 0:
        raise ValueError(f"Calibration dataset spec {spec!r} must include at least one text key.")
    repeat = _parse_bool(_last_query_value(query, "repeat", "oversample", "cycle"), default=False)

    if parsed.scheme in ("", "builtin"):
        dataset_id = (parsed.netloc + parsed.path).lstrip("/")
        if not dataset_id:
            raise ValueError(f"Missing built-in dataset name in spec: {spec!r}")
        return CalibDatasetSpec(
            kind="builtin",
            dataset_id=dataset_id,
            split=split,
            text_key=text_key,
            text_keys=text_keys,
            weight=weight,
            repeat=repeat,
        )

    if parsed.scheme == "hf":
        dataset_id = (parsed.netloc + parsed.path).lstrip("/")
        if not dataset_id:
            raise ValueError(f"Missing Hugging Face dataset path in spec: {spec!r}")
        return CalibDatasetSpec(
            kind="hf",
            dataset_id=dataset_id,
            config_name=_last_query_value(query, "name", "config"),
            split=split,
            text_key=text_key,
            text_keys=text_keys,
            weight=weight,
            revision=_last_query_value(query, "revision"),
            trust_remote_code=_parse_bool(_last_query_value(query, "trust_remote_code"), default=False),
            repeat=repeat,
        )

    raise ValueError(
        f"Unsupported calibration dataset spec {spec!r}. "
        "Use a built-in alias like 'wikitext2?weight=0.5' or "
        "'hf://org/dataset?split=train&text_key=text&weight=0.5'."
    )


def _allocate_mixed_sample_counts(specs: list[CalibDatasetSpec], n_samples: int) -> list[int]:
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    total_weight = sum(spec.weight for spec in specs)
    if total_weight <= 0:
        raise ValueError("At least one calibration dataset must have positive weight.")

    scaled = [spec.weight * n_samples / total_weight for spec in specs]
    counts = [math.floor(value) for value in scaled]
    remainder = n_samples - sum(counts)

    order = sorted(range(len(specs)), key=lambda i: (scaled[i] - counts[i], specs[i].weight), reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1

    return counts


def _load_dataset_from_spec(spec: CalibDatasetSpec, *, split: str, seed: int):
    if spec.kind == "builtin":
        data = spec.dataset_id
        if data == "pileval":
            if split != "validation":
                warnings.warn("The split argument is ignored when data is 'pileval'.")
            return load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(seed=seed)
        if data == "wikitext2":
            return load_dataset("wikitext", "wikitext-2-raw-v1", split=split).shuffle(seed=seed)
        if data == "c4":
            if split == "train":
                return load_dataset(
                    "allenai/c4",
                    data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                    split=split,
                ).shuffle(seed=seed)
            if split == "validation":
                return load_dataset(
                    "allenai/c4",
                    data_files={"validation": "en/c4-validation.00001-of-00008.json.gz"},
                    split=split,
                ).shuffle(seed=seed)
            raise ValueError(f"Invalid split for c4: {split}")
        if data == "redpajama":
            test_split, val_split = 0.2, 0.1
            dataset = load_dataset(
                "liang2kl/RedPajama-Data-1T-Sample-Backup",
                split="train",
                trust_remote_code=True,
            )
            dataset = dataset.shuffle(seed=seed)
            test_size = int(len(dataset) * test_split)
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - test_size - val_size
            if split == "test":
                return dataset.select(range(len(dataset) - test_size, len(dataset)))
            if split == "validation":
                return dataset.select(range(len(dataset) - test_size - val_size, len(dataset) - test_size))
            if split == "train":
                return dataset.select(range(0, train_size))
            raise ValueError(f"Invalid split: {split}")
        raise NotImplementedError(f"Unsupported built-in calibration dataset: {data}")

    if spec.kind == "hf":
        args = [spec.dataset_id]
        if spec.config_name is not None:
            args.append(spec.config_name)
        kwargs = {
            "split": split,
            "trust_remote_code": spec.trust_remote_code,
        }
        if spec.revision is not None:
            kwargs["revision"] = spec.revision
        return load_dataset(*args, **kwargs).shuffle(seed=seed)

    raise ValueError(f"Unsupported calibration dataset kind: {spec.kind}")


def get_blocks(model: nn.Module) -> nn.ModuleList:
    model_class_name = model.__class__.__name__
    if model_class_name in (
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForCausalLM",
    ):
        m = model.model
    else:
        raise NotImplementedError(type(model))

    return m.layers


_Linear_T = TypeVar("Linear", bound=nn.Module)


def get_named_linears(module: nn.Module, subclass: type[_Linear_T] = nn.Linear) -> dict[str, _Linear_T]:
    return {name: m for name, m in module.named_modules() if isinstance(m, subclass)}


def get_module_by_name(module, module_name):
    for name, m in module.named_modules():
        if name == module_name:
            return m
    return None


def set_module_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def load_model(
    model_path: str,
    device_map: str | None = None,
    dtype: torch.dtype | str | None = torch.float32,
    **kwargs,
) -> nn.Module:
    load_kwargs = dict(device_map=device_map, **kwargs)
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    return model


def load_tokenizer(model_path: str, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def move_embed(model, device):
    model_class_name = model.__class__.__name__
    if model_class_name in (
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForCausalLM",
    ):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    else:
        raise NotImplementedError(type(model))


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def get_mixed_calib_dataset(
    datasets: list[str],
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
) -> list[torch.Tensor]:
    specs = [parse_calib_dataset_spec(dataset, default_split=split) for dataset in datasets]
    counts = _allocate_mixed_sample_counts(specs, n_samples)
    results = []
    for spec, dataset_samples in zip(specs, counts):
        if dataset_samples == 0:
            continue
        results.extend(
            get_calib_dataset(
                data=spec,
                tokenizer=tokenizer,
                n_samples=dataset_samples,
                block_size=block_size,
                seed=seed,
                split=split,
            )
        )
    assert len(results) == n_samples, f"Expected {n_samples} samples, got {len(results)}"

    rand = random.Random(seed)
    rand.shuffle(results)

    return results


# Adapted from awq-llm
def get_calib_dataset(
    data: str | CalibDatasetSpec = "pileval",
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
) -> list[torch.Tensor]:
    spec = data if isinstance(data, CalibDatasetSpec) else parse_calib_dataset_spec(data, default_split=split)
    dataset = _load_dataset_from_spec(spec, split=spec.split or split, seed=seed)

    samples = []
    total_len = 0
    target_len = n_samples * block_size
    pass_idx = 0
    while total_len < target_len:
        pass_start_len = total_len
        current_dataset = dataset if pass_idx == 0 else dataset.shuffle(seed=seed + pass_idx)
        for row in current_dataset:
            missing_keys = [key for key in spec.text_keys if key not in row]
            if missing_keys:
                available = ", ".join(sorted(row.keys()))
                raise KeyError(
                    f"Columns {missing_keys!r} not found in calibration dataset {spec.dataset_id!r}. "
                    f"Available columns: {available}"
                )
            line_parts = []
            for key in spec.text_keys:
                value = row[key]
                if not isinstance(value, str):
                    value = str(value)
                value = value.strip()
                if value:
                    line_parts.append(value)
            if len(line_parts) == 0:
                continue
            line_encoded = tokenizer.encode("\n\n".join(line_parts))
            if len(line_encoded) > block_size:
                continue
            sample = torch.tensor([line_encoded])
            if sample.numel() == 0:
                continue
            samples.append(sample)
            total_len += len(line_encoded)
            if total_len >= target_len:
                break
        if total_len >= target_len or not spec.repeat:
            break
        if total_len == pass_start_len:
            raise ValueError(
                f"Unable to collect tokens from calibration dataset {spec.dataset_id!r} using keys {spec.text_keys!r}."
            )
        pass_idx += 1
    samples = torch.cat(samples, dim=1).squeeze(0)
    n_split = min(samples.shape[0] // block_size, n_samples)
    if n_split < n_samples:
        raise ValueError(
            f"Calibration dataset {spec.dataset_id!r} produced only {n_split} blocks, expected {n_samples}. "
            "Enable repeat=true to oversample short datasets or lower the requested sample count."
        )

    return [samples[i * block_size : (i + 1) * block_size] for i in range(n_split)]


@torch.no_grad()
def catch_first_layer_input(
    model: nn.Module,
    layers: nn.ModuleList,
    samples: torch.Tensor,
    batch_size: int | None,
) -> tuple[torch.Tensor, dict]:
    layer_kwargs = {}
    batched = batch_size is not None
    inps: list[torch.Tensor] = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            # Bypass __setattr__ of nn.Module
            object.__setattr__(self, "module", module)

        def forward(self, inp, **kwargs):
            inps.append(inp)
            if len(layer_kwargs) == 0:
                layer_kwargs.update(kwargs)
            raise ValueError

        def __getattr__(self, name):
            return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    batch_size = samples.shape[0] if not batched or batch_size <= 0 else batch_size
    num_batches = samples.shape[0] // batch_size
    samples_batch = samples.chunk(num_batches)
    for samples in samples_batch:
        try:
            model(samples.to(next(model.parameters()).device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    if not batched:
        inps = inps[0]

    layer_kwargs["use_cache"] = False
    if "past_key_value" in layer_kwargs:
        del layer_kwargs["past_key_value"]
    if "past_key_values" in layer_kwargs:
        del layer_kwargs["past_key_values"]

    return inps, layer_kwargs


class CachedTensorShards:
    def __init__(
        self,
        batches: list[torch.Tensor],
        num_shards: int,
        *,
        target_device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
    ):
        assert len(batches) % num_shards == 0
        if batches[0].device != offload_device:
            self.batches = [b.to(offload_device) for b in batches]
        else:
            self.batches = batches
        self.num_shards = num_shards
        self.current_shard: int = None
        self.cached_shard: list[torch.Tensor] = None
        self.target_device = target_device

    def _switch_shard(self, shard_index: int) -> None:
        if self.current_shard == shard_index:
            return
        self.current_shard = shard_index
        start, end = self._get_shard_range(shard_index)
        self.cached_shard = self.batches[start:end]
        self.cached_shard = [b.to(self.target_device) for b in self.cached_shard]

    def _get_shard_range(self, index: int) -> tuple[int, int]:
        if self.num_shards == 1:
            return 0, len(self.batches)
        shard_size = len(self.batches) // self.num_shards
        start = shard_size * index
        if index == self.num_shards - 1:
            end = len(self.batches)
        else:
            end = shard_size * (index + 1)
        return start, end

    def __getitem__(self, index: int) -> torch.Tensor:
        shard_len = len(self.batches) // self.num_shards
        shard_index = index // shard_len
        if self.current_shard != shard_index:
            self._switch_shard(shard_index)
        offset = index % shard_len
        return self.cached_shard[offset]

    def __iter__(self) -> "Iterator":
        return self.Iterator(self)

    def __len__(self) -> int:
        return len(self.batches)

    class Iterator:
        def __init__(self, batches: "CachedTensorShards"):
            self.batches = batches
            self.current_index = 0

        def __iter__(self):
            return self

        def __next__(self) -> torch.Tensor:
            if self.current_index >= len(self.batches):
                raise StopIteration
            result = self.batches[self.current_index]
            self.current_index += 1
            return result

        def __len__(self) -> int:
            return len(self.batches)


class CosineAnnealingParam:
    def __init__(self, start_value: float, end_value: float, T_max: int):
        """
        Args:
            start_value (float): The initial value (equivalent to eta_max).
            end_value (float): The final value (equivalent to eta_min).
            T_max (int): Maximum number of steps.
        """
        self.start_value = start_value
        self.end_value = end_value
        self.T_max = T_max
        self._step = -1

    def step(self) -> float:
        self._step += 1

        if self._step >= self.T_max:
            return self.end_value

        cos_val = math.cos(math.pi * self._step / self.T_max)
        return self.end_value + (self.start_value - self.end_value) * (1 + cos_val) / 2


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


logger = get_logger("ParoQuant")
