from __future__ import annotations

import unittest

import torch

from paroquant.kernels.cuda import scaled_pairwise_rotation


GROUP_SIZE = 128
KROT = 8
HIDDEN_SIZE = 128
BATCH = 3


def _build_rotation_inputs(
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    idx_rows = []
    theta_rows = []
    for _ in range(KROT):
        perm = torch.randperm(HIDDEN_SIZE, generator=gen, dtype=torch.int32)
        idx_rows.append(perm.to(torch.int16))
        theta_rows.append(torch.randn(HIDDEN_SIZE // 2, generator=gen, dtype=torch.float32) * 0.05)

    idx_ij = torch.stack(idx_rows, dim=0).to(device=device)
    theta = torch.stack(theta_rows, dim=0).to(device=device, dtype=dtype)
    scales = (0.5 + torch.rand(HIDDEN_SIZE, generator=gen, dtype=torch.float32)).to(device=device, dtype=dtype)
    return idx_ij, theta, scales


def _reference_scaled_pairwise_rotation(
    x: torch.Tensor,
    idx_ij: torch.Tensor,
    theta: torch.Tensor,
    scales: torch.Tensor | None = None,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    out = x.float()
    if scales is not None:
        out = (out * scales.float()).to(dtype=x.dtype).float()

    krot, hidden_size = idx_ij.shape
    num_groups = hidden_size // group_size

    for r in range(krot):
        row_idx = idx_ij[r].to(torch.long)
        row_theta = theta[r].float()
        next_out = out.clone()
        theta_ptr = 0
        for g in range(num_groups):
            group_offset = g * group_size
            group_idx = row_idx[group_offset : group_offset + group_size]
            for p in range(group_size // 2):
                local_i = group_idx[2 * p].item()
                local_j = group_idx[2 * p + 1].item()
                i = group_offset + local_i
                j = group_offset + local_j
                angle = row_theta[theta_ptr]
                theta_ptr += 1

                c = torch.cos(angle)
                s = torch.sin(angle)
                xi = out[:, i]
                xj = out[:, j]
                next_out[:, i] = c * xi + s * xj
                next_out[:, j] = c * xj - s * xi
        out = next_out.to(dtype=x.dtype).float()

    return out.to(dtype=x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for rotation kernel tests")
class RotationKernelTest(unittest.TestCase):
    device = torch.device("cuda")

    def _assert_forward_matches_reference(self, dtype: torch.dtype, *, atol: float, rtol: float) -> None:
        idx_ij, theta, scales = _build_rotation_inputs(device=self.device, dtype=dtype)
        x = torch.randn(BATCH, HIDDEN_SIZE, device=self.device, dtype=dtype)

        actual = scaled_pairwise_rotation(x, idx_ij, theta, scales, GROUP_SIZE)
        expected = _reference_scaled_pairwise_rotation(x, idx_ij, theta, scales, GROUP_SIZE)

        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)

    def _assert_backward_finite(self, dtype: torch.dtype) -> None:
        idx_ij, theta, scales = _build_rotation_inputs(device=self.device, dtype=dtype, seed=1)

        x = torch.randn(BATCH, HIDDEN_SIZE, device=self.device, dtype=dtype, requires_grad=True)
        theta = theta.detach().clone().requires_grad_(True)
        scales = scales.detach().clone().requires_grad_(True)
        grad = torch.randn(BATCH, HIDDEN_SIZE, device=self.device, dtype=dtype)

        out = scaled_pairwise_rotation(x, idx_ij, theta, scales, GROUP_SIZE)
        loss = (out * grad).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(theta.grad)
        self.assertIsNotNone(scales.grad)
        self.assertTrue(torch.isfinite(x.grad.float()).all().item())
        self.assertTrue(torch.isfinite(theta.grad.float()).all().item())
        self.assertTrue(torch.isfinite(scales.grad.float()).all().item())

    def test_forward_matches_reference_fp16(self) -> None:
        self._assert_forward_matches_reference(torch.float16, atol=2e-3, rtol=2e-3)

    def test_backward_produces_finite_grads_fp16(self) -> None:
        self._assert_backward_finite(torch.float16)

    @unittest.skipUnless(torch.cuda.is_bf16_supported(), "CUDA BF16 is not supported on this GPU")
    def test_forward_matches_reference_bf16(self) -> None:
        self._assert_forward_matches_reference(torch.bfloat16, atol=1e-2, rtol=1e-2)

    @unittest.skipUnless(torch.cuda.is_bf16_supported(), "CUDA BF16 is not supported on this GPU")
    def test_backward_produces_finite_grads_bf16(self) -> None:
        self._assert_backward_finite(torch.bfloat16)
