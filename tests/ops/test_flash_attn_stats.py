import math
import pytest
import torch


def _ref_stats(q: torch.Tensor, k: torch.Tensor, scale: float):
    """Reference computation of per-row softmax statistics.

    Args:
        q: (B, Sq, H, D)
        k: (B, Sk, H, D)
        scale: scaling factor applied to the matmul (usually 1/sqrt(D)).

    Returns:
        row_max: (B, H, Sq)  – max logits per row
        sum_exp: (B, H, Sq)  – Σ exp(logits – row_max)
        lse:     (B, H, Sq)  – log-sum-exp = row_max + log(sum_exp)
    """
    # Bring head dim forward for convenience: (B, H, Sq, D)
    q_ = q.permute(0, 2, 1, 3).contiguous()
    k_ = k.permute(0, 2, 3, 1).contiguous()  # (B, H, D, Sk)

    # Kernel stores the *un-scaled* max (before multiplying by `scale`).
    logits_raw = torch.matmul(q_, k_)          # (B, H, Sq, Sk)

    # Per-row maximum before scaling
    row_max = logits_raw.max(dim=-1).values    # (B, H, Sq)

    # The softmax itself is computed on the scaled logits
    logits_scaled = logits_raw * scale

    # exp(logits_scaled − row_max * scale)
    sum_exp = torch.exp(logits_scaled - row_max.unsqueeze(-1) * scale).sum(dim=-1)

    # log-sum-exp that the kernel writes:  row_max*scale + log(sum_exp)
    lse = row_max * scale + torch.log(sum_exp)
    return row_max, sum_exp, lse


def test_flash_attn_emits_stats_forward():
    torch.manual_seed(0)
    B, Sq, Sk, H, D = 2, 8, 8, 2, 32
    scale = 1.0 / math.sqrt(D)

    device = "cuda"
    dtype = torch.float16

    q = torch.randn(B, Sq, H, D, device=device, dtype=dtype)
    k = torch.randn(B, Sk, H, D, device=device, dtype=dtype)
    v = torch.randn(B, Sk, H, D, device=device, dtype=dtype)

    import flash_attn_2_cuda as fa

    # Call the low-level kernel directly.  Signature:
    # fwd(q, k, v, out, alibi, p_dropout, softmax_scale, causal,
    #     win_left, win_right, softcap, return_softmax, gen)
    out, lse, row_max, sum_exp, *_ = fa.fwd(
        q, k, v,
        None,       # out tensor (let kernel allocate)
        None,       # alibi slopes
        0.0,        # dropout
        scale,
        False,      # causal
        -1, -1,     # window sizes
        0.0,        # softcap
        False,      # return_softmax
        None,       # RNG
    )

    # Reference stats in FP32 for accuracy
    ref_row_max, ref_sum_exp, ref_lse = _ref_stats(q.float(), k.float(), scale)

    # Use float32 for comparison to avoid half-precision noise
    assert torch.allclose(row_max.float(), ref_row_max, atol=1e-3, rtol=1e-3)
    assert torch.allclose(sum_exp.float(), ref_sum_exp, atol=1e-2, rtol=1e-2)
    assert torch.allclose(lse.float(), ref_lse, atol=1e-3, rtol=1e-3)

    # Ensure shapes match expectations
    assert row_max.shape == (B, H, Sq)
    assert sum_exp.shape == (B, H, Sq)
    assert lse.shape == (B, H, Sq)
    assert out.shape == (B, Sq, H, D) 