# Copyright (c) 2026, Ducc (MetaHarness4Attn)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Dual-State Delta Rule — fused recurrent kernel for two-state DeltaNet.
#
# This is a **composite kernel** that expands compressed k/v to full temporal
# resolution and delegates to the existing fused_recurrent_delta_rule.
#
# Problem
# -------
# Dual-State DeltaNet has:
#   - q: [B, T, H, K]   — per-token queries (full T resolution)
#   - k: [B, N, H, K]   — compressed keys  (N = ceil(T/r), r = scale_ratio)
#   - v: [B, N, H, V]   — compressed values
#   - beta: [B, N, H]   — per-block beta
#
# The semantics are:
#   - READ:  every token t reads S_aux @ q[t]       (per-token, T times)
#   - WRITE: at block boundaries, S_aux += k̄·(v̄ - S@k̄)  (per-block, N times)
#
# This is mathematically equivalent to running fused_recurrent_delta_rule
# with k/v/beta expanded to full resolution where:
#   - k[t] = 0, v[t] = 0, beta[t] = 0  for non-boundary positions
#   - k[t] = k̄[b], v[t] = v̄[b], beta[t] = β̄[b]  at boundary positions
#
# At positions where k=0 and beta=0, the delta rule write is a no-op
# (S += 0 ⊗ (0 - S@0) = 0), but the READ still works (o = S @ q).
# This gives the correct "read per-token, write per-block" behavior.
#
# Backward is handled automatically by autograd through the existing kernel,
# so no custom Triton backward is needed.

from __future__ import annotations

import torch

from fla.ops.delta_rule import fused_recurrent_delta_rule


def fused_recurrent_ds_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale_ratio: int,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Fused recurrent dual-state delta rule.

    Takes full-resolution queries q [B, T, H, K] and compressed
    k [B, N, H, K], v [B, N, H, V] with N = ceil(T/scale_ratio).
    Expands k/v to full resolution with writes at block boundaries,
    then delegates to ``fused_recurrent_delta_rule``.

    Args:
        q: Queries of shape ``[B, T, H, K]`` — full resolution.
        k: Compressed keys of shape ``[B, N, H, K]`` where N = ceil(T/r).
        v: Compressed values of shape ``[B, N, H, V]``.
        beta: Per-block beta of shape ``[B, N, H]``.
        scale_ratio: L-axis compression ratio (r). A compressed write
            is performed every r tokens, at position r-1, 2r-1, ...
        scale: Scale factor. If None, defaults to ``1 / sqrt(K)``.
        initial_state: Optional initial state ``[N, H, K, V]``. For the
            aux path this is typically None (zero-initialized internally).
        output_final_state: Whether to return the final state.
        use_qk_l2norm_in_kernel: Apply L2 norm inside the kernel.
        cu_seqlens: Variable-length sequence boundaries.

    Returns:
        o: Output of shape ``[B, T, H, V]``.
        final_state: Final state if ``output_final_state=True``, else None.
    """
    B, T, H, K = q.shape
    N = k.shape[1]
    V = v.shape[-1]
    r = scale_ratio
    device = q.device
    dtype = q.dtype

    # ------------------------------------------------------------------
    # Vectorized expansion: scatter compressed k/v into full T positions
    #
    # Write positions are at the END of each block:
    #   block 0 → position r-1
    #   block 1 → position 2r-1
    #   ...
    #   block N-1 → position min(N*r-1, T-1)
    #
    # At all other positions, k=0, v=0, beta=0 → no delta write.
    # ------------------------------------------------------------------
    # [N] tensor of write positions, clamped to T-1 for last partial block
    write_pos = torch.arange(N, device=device) * r + (r - 1)
    write_pos = write_pos.clamp(max=T - 1)

    k_full = torch.zeros(B, T, H, K, device=device, dtype=dtype)
    v_full = torch.zeros(B, T, H, V, device=device, dtype=dtype)
    beta_full = torch.zeros(B, T, H, device=device, dtype=dtype)

    # Scatter: k[:, write_pos] → k_full
    # Both operands have shape [B, N, H, ...]
    k_full[:, write_pos] = k
    v_full[:, write_pos] = v
    beta_full[:, write_pos] = beta

    # ------------------------------------------------------------------
    # Delegate to the standard fused_recurrent_delta_rule
    #
    # The existing kernel's per-token loop now naturally does:
    #   t=0..r-2: k[t]=0, v[t]=0, beta[t]=0 → no write, but o[t]=S@q[t]
    #   t=r-1:    k[t]=k̄, v[t]=v̄, beta[t]=β̄ → delta write AFTER read
    #   t=r..2r-2: no write, reads from post-write state
    #   ...
    #
    # This exactly matches the block-by-block semantics.
    # ------------------------------------------------------------------
    o, final_state = fused_recurrent_delta_rule(
        q=q,
        k=k_full,
        v=v_full,
        beta=beta_full,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )

    return o, final_state