"""
CLA – Compressed Linear Attention.

Fuses CSA's compress+indexer mechanism with DeltaNet's linear (delta-rule)
attention, creating an architecture with:
  - Sparse compression for sequence reduction
  - Linear attention (O(d²) recurrent state) for efficient long-range modeling

3-Pathway Design:
  Pathway 1: Sliding Window (LocalAttention + RoPE)
  Pathway 2: Compressed Linear Attention (avg-pool Q/K/V → DeltaNet → expand)
  Pathway 3: Selected Linear Attention (Indexer top-k → gather DeltaNet output → weighted sum)

Output = g1*out1 + g2*out2 + g3*out3 → out_proj
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

# from zoology.mixers.csa_hca import GatedPoolCompressor
from zoology.mixers.delta_net import DeltaNet
from zoology.mixers.slide_attn import SlidingAttn

# ---------------------------------------------------------------------------
# gated-pooling compressor
# ---------------------------------------------------------------------------

class LenGatedPoolCompressor(nn.Module):
    """Compresses a sequence via learned gated pooling over ``compress_ratio``
    consecutive tokens.

    When *overlap* is True (ratio <= 4), adjacent compression windows share one
    token for smoother boundaries.  This mirrors ``model.py:Compressor`` without
    quantization, RoPE, Hadamard rotation, or incremental-decode state.
    """

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        compress_ratio: int = 4,
        overlap: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = overlap and compress_ratio <= 4
        coff = 2 if self.overlap else 1

        self.wkv = nn.Linear(d_model, coff * head_dim, bias=False)
        self.wgate = nn.Linear(d_model, coff * head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim))
        self.norm = nn.RMSNorm(head_dim)

        nn.init.normal_(self.ape, mean=0.0, std=0.02)

        self.tau = nn.Parameter(torch.ones(1) * 2.0)   # 可学习温度
        self.base_weight = nn.Parameter(torch.ones(compress_ratio, 1) / compress_ratio)  # 可学习基值

    # ------------------------------------------------------------------
    def _overlap_transform(self, x: torch.Tensor, fill_value: float):
        """Shift the first half of dims one block forward (overlapping windows)."""
        b, n, r, coff_d = x.shape
        d = self.head_dim
        new = x.new_full((b, n, 2 * r, d), fill_value)
        new[:, :, r:, :] = x[:, :, :, d:]        # normal half
        new[:, 1:, :r, :] = x[:, :-1, :, :d]     # overlapping half (shifted)
        return new

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [b, s, d_model]  ->  compressed: [b, n_blocks, head_dim]"""
        b, s, _ = x.shape
        ratio = self.compress_ratio

        # pad to multiple of ratio
        remainder = s % ratio
        pad_len = (ratio - remainder) % ratio
        if remainder > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        kv = self.wkv(x).unflatten(1, (-1, ratio))
        # score = self.wgate(x).unflatten(1, (-1, ratio)) + self.ape

        base = self.base_weight.view(1,1,ratio,1)   # [1,1,ratio,1]
        score = base + (self.wgate(x).unflatten(1, (-1, ratio)) + self.ape) / self.tau

        if pad_len > 0:
            valid_mask = torch.zeros(b, s + pad_len, device=x.device, dtype=torch.bool)
            valid_mask[:, :s] = True
            score = score.masked_fill(~valid_mask.view(b, -1, ratio, 1), float("-inf"))

        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)
            score = self._overlap_transform(score, float("-inf"))

        kv = (kv * score.softmax(dim=2)).sum(dim=2)          # [b, n_blocks, head_dim]
        kv = self.norm(kv)

        return kv


class LenGatedPoolExpander(nn.Module):
    """
    Expands sequence length by factor `expand_ratio` using gated linear projection.
    Designed as the inverse of LenGatedPoolCompressor.

    When overlap=True (and expand_ratio <= 4), each input token produces both a normal
    window and an overlapping window. The overlapping window is shifted right and added
    to the next token's normal window, creating smooth transitions.

    Args:
        d_model: input/output feature dimension
        expand_ratio: integer factor to expand sequence length
        overlap: if True, enable overlapping expansion (only for ratio <= 4)
    """
    def __init__(self, d_model: int, expand_ratio: int, overlap: bool = False, **kwargs):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.overlap = overlap and expand_ratio <= 4
        coff = 2 if self.overlap else 1
        self.coff = coff

        # Learnable bias for each expanded position (shared across all tokens)
        self.ape = nn.Parameter(torch.empty(coff * expand_ratio, d_model))
        nn.init.normal_(self.ape, mean=0.0, std=0.02)

        # learnable base weight + temperature (mirrors LenGatedPoolCompressor)
        self.base_weight = nn.Parameter(torch.ones(coff * expand_ratio, 1) / (coff * expand_ratio))
        self.tau = nn.Parameter(torch.ones(1) * 2.0)

    def _overlap_combine(self, x: torch.Tensor):
        """
        Combine normal and overlapping windows into final expanded sequence.

        x: [B, S, 2*ratio, D] where ratio = expand_ratio
        Returns: [B, S*ratio, D]
        """
        B, S, _, D = x.shape
        ratio = self.expand_ratio
        normal = x[:, :, :ratio, :]      # [B, S, ratio, D]
        overlap = x[:, :, ratio:, :]     # [B, S, ratio, D]

        out = torch.zeros(B, S * ratio, D, device=x.device, dtype=x.dtype)
        out = out.view(B, S, ratio, D)
        # Normal part occupies its own block
        out = out + normal
        # Overlapping part from previous block is added to the beginning of current block
        out[:, 1:, :, :] += overlap[:, :-1, :, :]
        out = out.view(B, -1, D)
        return out

    def forward(self, x):
        """
        x: [B, S, D]
        Returns: [B, S * expand_ratio, D]
        """
        B, S, D = x.shape
        ratio = self.expand_ratio
        coff = self.coff

        # Project to raw expansion and gates (fused)
        expand = self.wexpand(x).view(B, S, coff * ratio, D)       # [B, S, coff*R, D]
        raw_gate = self.wgate(x).view(B, S, coff * ratio, D)       # [B, S, coff*R, D]

        # Gating: base_weight + (raw_gate + ape) / tau  → sigmoid
        base = self.base_weight.view(1, 1, coff * ratio, 1)
        gate = torch.sigmoid(
            base + (raw_gate + self.ape.view(1, 1, coff * ratio, D)) / self.tau
        )
        out = expand * gate                                         # [B, S, coff*R, D]

        if self.overlap:
            out = self._overlap_combine(out)    # [B, S*ratio, D]
        else:
            out = out.view(B, -1, D)            # [B, S*ratio, D]

        return out


class CompressedLinearAttention(nn.Module):
    def __init__(
            self, 
            d_model: int,
            num_heads: int,
            use_sliding: bool = False,
            compress_ratio: int = 4,
            overlap: bool = False,
            window_size: int = 32,
            delta_net_kwargs: dict = None,
            **kwargs,
        ):
        super().__init__()
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.use_sliding = use_sliding
        self.compress_ratio = compress_ratio

        self.sliding_attn = SlidingAttn(d_model, block_size=window_size)
        self.compressor = LenGatedPoolCompressor(
            d_model=d_model,
            head_dim=self.head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
        # project compressed head_dim → d_model before DeltaNet
        self.pre_proj = nn.Linear(self.head_dim, d_model, bias=False)
        # DeltaNet operates at full d_model — state capacity matches baseline
        delta_net_kwargs = delta_net_kwargs or {}
        self.delta_net = DeltaNet(d_model=d_model, **delta_net_kwargs)

        # skip connection: lightweight MLP processes per-token residual
        # (token - block_center) that was lost during R→1 compression.
        # zero-init → at init output = 0, behaving exactly like repeat baseline.
        self.skip_mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )
        nn.init.zeros_(self.skip_mlp[-1].weight)
        nn.init.zeros_(self.skip_mlp[-1].bias)

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        R = self.compress_ratio

        if self.use_sliding:
            x = self.sliding_attn(x) + x  # residual

        # pad to multiple of R (matching compressor's internal padding)
        remainder = S % R
        pad_len = (R - remainder) % R
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x
        x_blocks = x_padded.view(B, -1, R, D)          # [B, n_blocks, R, D]

        compressed = self.compressor(x)                 # [B, n_blocks, head_dim]
        # project to d_model — this is the "block center"
        dn_input = self.pre_proj(compressed)            # [B, n_blocks, D]
        dn = self.delta_net(dn_input)                   # [B, n_blocks, D]

        # Path 1: repeat baseline (shared block center)
        base = dn.unsqueeze(2).expand(-1, -1, R, -1)   # [B, n_blocks, R, D]

        # Path 2: skip connection — residual from pre-compression tokens
        # r_i = token_i - block_center → captures info lost in pooling
        center = dn_input.unsqueeze(2)                  # [B, n_blocks, 1, D]
        residual = x_blocks - center                    # [B, n_blocks, R, D]
        skip_info = self.skip_mlp(residual)             # [B, n_blocks, R, D]

        out = (base + skip_info).view(B, -1, D)[:, :S, :]
        return out

    # ------------------------------------------------------------------
    def state_size(self, sequence_length: int = 2048) -> int:
        """Compute total state size (for memory accounting).

        Returns:
            State size = delta_net.state_size
        """
        return self.delta_net.state_size(sequence_length)


class ExpandLinearAttention(nn.Module):
    """
    Expands sequence length, processes with DeltaNet, then folds back.

    Flow::

        x → [optional SlidingAttn] → LenGatedPoolExpander → DeltaNet
          → reshape (concat over R) → out_proj

    **Rationale.**  The original ``CompressedLinearAttention`` reduces sequence
    length *before* DeltaNet, creating an information bottleneck that hurts
    length generalisation.  This module does the opposite: it uses
    :class:`LenGatedPoolExpander` to produce ``expand_ratio`` sub-tokens from each
    input token via learned gated projections.  DeltaNet therefore has *more*
    recurrent updates — and thus more capacity to encode the input — without
    any lossy front-end compression.

    After DeltaNet, the expanded outputs **are not pooled** (no avg/last/gated
    selection that would discard information).  Instead, the ``R`` outputs per
    group are **concatenated** along the feature dimension (``R·D``) and then
    linearly projected back to ``D``.  This preserves all the fine-grained
    information that DeltaNet produced, at the cost of a slightly larger
    projection matrix.

    Args:
        d_model: input/output feature dimension.
        num_heads: number of attention heads (used for DeltaNet).
        expand_ratio: factor by which to expand the sequence length.
        overlap: whether to use overlapping expansion windows.
        use_sliding: if True, apply a sliding-window attention as a
            preprocessing step.
        window_size: sliding-window size.
        delta_net_kwargs: keyword arguments forwarded to :class:`DeltaNet`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        expand_ratio: int = 2,
        overlap: bool = False,
        use_sliding: bool = False,
        window_size: int = 32,
        delta_net_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.expand_ratio = expand_ratio
        self.use_sliding = use_sliding

        # optional sliding-window preprocessing
        if use_sliding:
            self.sliding_attn = SlidingAttn(d_model, block_size=window_size)
        else:
            self.sliding_attn = None

        # gated expander
        self.expander = LenGatedPoolExpander(
            d_model=d_model,
            expand_ratio=expand_ratio,
            overlap=overlap,
        )

        # DeltaNet — operates at full d_model (no head_dim bottleneck)
        delta_net_kwargs = delta_net_kwargs or {}
        delta_net_kwargs.setdefault('num_heads', num_heads)
        self.delta_net = DeltaNet(d_model=d_model, **delta_net_kwargs)

        # Concat-and-project: preserve all R×D dimensions → learn projection to D
        self.out_proj = nn.Linear(expand_ratio * d_model, d_model, bias=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D]  ->  out: [B, S, D]"""
        B, S, D = x.shape
        R = self.expand_ratio

        if self.use_sliding and self.sliding_attn is not None:
            x = self.sliding_attn(x) + x

        # 1) expand: [B, S, D] -> [B, S*R, D]
        expanded = self.expander(x)

        # 2) DeltaNet processes the longer sequence
        dn_out = self.delta_net(expanded)               # [B, S*R, D]

        # 3) concat over R → project back to D  (no information loss)
        dn_out = dn_out[:, :S * R, :]                   # safety trim
        out = dn_out.view(B, S, R * D)                  # [B, S, R*D]
        out = self.out_proj(out)                        # [B, S, D]

        return out

    # ------------------------------------------------------------------
    def state_size(self, sequence_length: int = 2048) -> int:
        return self.delta_net.state_size(sequence_length)


class MultiHeadCLA(nn.Module):
    def __init__(
            self, 
            d_model: int,
            num_heads: int,
            use_sliding: bool = False,
            compress_ratios: List[int] = [2, 4],
            overlap: bool = False,
            window_size: int = 32,
            delta_net_kwargs: dict = None,
            fusion: str = 'sum',   # 'sum', 'learned_sum', 'concat'
            **kwargs,
        ):
        super().__init__()
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.use_sliding = use_sliding
        self.compress_ratios = compress_ratios
        self.fusion = fusion

        if use_sliding:
            self.sliding_attn = SlidingAttn(d_model, block_size=window_size)
        else:
            self.sliding_attn = None
        
        self.compressors = nn.ModuleList()
        self.block_projs = nn.ModuleList()
        self.out_projs = nn.ModuleList()
        self.block_pos_embs = nn.ModuleList()

        for ratio in compress_ratios:
            self.compressors.append(
                LenGatedPoolCompressor(
                    d_model=d_model,
                    head_dim=self.head_dim,
                    compress_ratio=ratio,
                    overlap=overlap,
                )
            )
            self.block_projs.append(nn.Linear(self.head_dim, d_model))
            # 上采样：d_model -> d_model * ratio
            self.out_projs.append(nn.Linear(d_model, d_model * ratio))
            # 块内位置编码（可学习）
            self.block_pos_embs.append(nn.Embedding(ratio, d_model))
        
        delta_net_kwargs = delta_net_kwargs or {}
        delta_net_kwargs.setdefault('d_model', d_model)
        self.delta_net = DeltaNet(**delta_net_kwargs)
        
        # 融合层（如果使用 concat 或 learned_sum）
        if fusion == 'concat':
            self.fusion_proj = nn.Linear(len(compress_ratios) * d_model, d_model)
        elif fusion == 'learned_sum':
            self.fusion_weights = nn.Parameter(torch.ones(len(compress_ratios)) / len(compress_ratios))
        else:
            self.fusion_weights = None
            self.fusion_proj = None

    def forward(self, x):
        # x: [B, S, D]
        if self.use_sliding:
            x = self.sliding_attn(x) + x  # residual
        
        outputs = []
        for i, ratio in enumerate(self.compress_ratios):
            # 1. 压缩
            compressed = self.compressors[i](x)                 # [B, n_blocks, head_dim]
            compressed = self.block_projs[i](compressed)        # [B, n_blocks, D]
            # 2. 共享 DeltaNet 处理压缩序列
            dn = self.delta_net(compressed)                     # [B, n_blocks, D]
            # 3. 上采样 + 块内位置编码
            dn_expanded = dn.unsqueeze(2).expand(-1, -1, ratio, -1)  # [B, n_blocks, ratio, D]
            pos_emb = self.block_pos_embs[i].weight.unsqueeze(0).unsqueeze(0)  # [1,1,ratio,D]
            out_branch = dn_expanded + pos_emb
            out_branch = out_branch.view(out_branch.size(0), -1, self.d_model)  # [B, n_blocks*ratio, D]
            out_branch = out_branch[:, :x.shape[1], :]           # 截断至原始长度
            outputs.append(out_branch)

        # 融合多分支
        if self.fusion == 'sum':
            out = torch.stack(outputs, dim=0).sum(dim=0)
        elif self.fusion == 'learned_sum':
            weights = torch.softmax(self.fusion_weights, dim=0)
            out = sum(w * o for w, o in zip(weights, outputs))
        elif self.fusion == 'concat':
            out = torch.cat(outputs, dim=-1)
            out = self.fusion_proj(out)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
        return out

    # ------------------------------------------------------------------
    def state_size(self, sequence_length: int = 2048) -> int:
        """Compute total state size (for memory accounting).

        Returns:
            State size = delta_net.state_size
        """
        return self.delta_net.state_size(sequence_length)


# ---------------------------------------------------------------------------
# Old version of Multi-scale DeltaNet — parallel states at different feature resolutions
# ---------------------------------------------------------------------------
class OldMultiScaleDeltaNet(nn.Module):
    """
    Multi-scale DeltaNet with auxiliary low-dimensional state.

    Rationale
    ---------
    DeepSeek-V4's CSA augments softmax attention by providing *additional*
    compressed KV positions for the query to attend to.  Linear attention
    (DeltaNet) cannot do this — its Q/K/V must be length-aligned.  The
    analogous operation in feature-space is to provide *additional* state
    capacity at a reduced feature resolution:

        main:  S_main ∈ R^{D×D}          (full resolution, per token)
        aux:   S_aux  ∈ R^{D/R × D/R}    (compressed, per token)

    Each token updates **both** states.  The aux state has (1/R²) the
    capacity of the main state, but it processes the same S tokens —
    capturing lower-rank / coarser structure that complements the main
    state's fine-grained representation.

    Flow
    ----
        x ──→ main DeltaNet ──→ o_main ─┐
          └→ down_proj → aux DeltaNet → up_proj(aux) → + → output

    Args:
        d_model: input/output feature dimension.
        num_heads: number of attention heads for the *main* DeltaNet.
        scale_ratio: feature-dim compression ratio for the auxiliary state.
        delta_net_kwargs: keyword arguments forwarded to **both** DeltaNets.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        scale_ratio: int = 2,
        **kwargs,  # forwarded to BOTH DeltaNets
    ):
        super().__init__()
        self.d_model = d_model
        self.scale_ratio = scale_ratio

        # main state — full resolution
        self.main_net = DeltaNet(d_model=d_model, num_heads=num_heads, **kwargs)

        # aux state — compressed feature dimension
        aux_d_model = d_model // scale_ratio
        aux_num_heads = max(1, num_heads // scale_ratio)

        self.aux_down = nn.Linear(d_model, aux_d_model, bias=False)
        self.aux_net = DeltaNet(d_model=aux_d_model, num_heads=aux_num_heads, **kwargs)
        self.aux_up = nn.Linear(aux_d_model, d_model, bias=False)

        # zero-init aux_up → at init, aux path contributes nothing
        nn.init.zeros_(self.aux_up.weight)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D]  ->  out: [B, S, D]"""
        # main: full-resolution processing
        o_main = self.main_net(x)         # [B, S, D]

        # aux: compress → DeltaNet → expand
        aux_x = self.aux_down(x)          # [B, S, D/R]
        o_aux = self.aux_net(aux_x)       # [B, S, D/R]
        o_aux = self.aux_up(o_aux)        # [B, S, D]

        return o_main + o_aux

    # ------------------------------------------------------------------
    def state_size(self, sequence_length: int = 2048) -> int:
        return (
            self.main_net.state_size(sequence_length)
            + self.aux_net.state_size(sequence_length)
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 配置参数
    d_model = 64
    num_heads = 4                 # 确保 d_model % num_heads == 0
    compress_ratio = 4
    window_size = 8             # 滑动窗口大小
    batch_size = 2
    seq_len = 64                # 序列长度（能被 compress_ratio 整除）

    # DeltaNet 参数（轻量设置）— 移除 num_heads 以避免与顶层参数冲突
    delta_kwargs = {
        "expand_k": 1.0,
        "expand_v": 1.0,
        "num_heads": 4,
        "use_beta": True,
        "use_gate": False,
        "use_short_conv": True,
        "conv_size": 4,
        "conv_bias": False,
        "allow_neg_eigval": False,
        "qk_activation": "silu",
        "qk_norm": "l2",
        "norm_eps": 1e-5,
    }

    # 实例化模型
    model = CompressedLinearAttention(
        d_model=d_model,
        num_heads=num_heads,
        compress_ratio=compress_ratio,
        window_size=window_size,
        delta_net_kwargs=delta_kwargs,
    ).to(device)
    model.train()   # 启用 dropout 等

    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # 1. 形状测试
    out = model(x)
    print(f"[Shape] Output: {out.shape}, expected: ({batch_size}, {seq_len}, {d_model})")
    assert out.shape == (batch_size, seq_len, d_model), f"Shape mismatch: {out.shape}"

    # 2. 数值稳定性测试
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    # 3. 梯度测试
    x_grad = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    loss = model(x_grad).sum()
    loss.backward()
    # 检查关键参数梯度是否存在
    assert model.pre_proj.weight.grad is not None, "pre_proj weight gradient missing"
    assert model.delta_net.q_proj.weight.grad is not None, "DeltaNet q_proj gradient missing"
    print("[Gradient] Passed")

    # 4. 可选：测试禁用滑动窗口
    if hasattr(model, 'use_sliding'):
        model.use_sliding = False
        out_no_sliding = model(x)
        assert out_no_sliding.shape == (batch_size, seq_len, d_model), "Disabled sliding: shape mismatch"
        print("[Sliding disabled] Works")

    # 5. 测试非整数倍序列长度（padding + 截断）
    seq_len_irregular = 63
    x_irregular = torch.randn(batch_size, seq_len_irregular, d_model, device=device)
    out_irregular = model(x_irregular)
    assert out_irregular.shape == (batch_size, seq_len_irregular, d_model), f"Irregular length shape: {out_irregular.shape}"
    print(f"[Irregular length {seq_len_irregular}] Output shape correct")

    print("All tests passed for CompressedLinearAttention.")

    # ------------------------------------------------------------------
    # Tests for ExpandLinearAttention
    # ------------------------------------------------------------------
    print("\n--- Testing ExpandLinearAttention (concat+project) ---")
    expand_ratio = 2

    model_ela = ExpandLinearAttention(
        d_model=d_model,
        num_heads=num_heads,
        expand_ratio=expand_ratio,
        overlap=False,
        use_sliding=False,
        delta_net_kwargs=delta_kwargs,
    ).to(device)

    # 1. Shape test
    out_ela = model_ela(x)
    print(f"  [Shape] {out_ela.shape}, expected ({batch_size}, {seq_len}, {d_model})")
    assert out_ela.shape == (batch_size, seq_len, d_model), f"Shape mismatch: {out_ela.shape}"

    # 2. Numerical stability
    assert torch.isfinite(out_ela).all(), "Output contains NaN or Inf"
    print("  [Finite] OK")

    # 3. Gradient test
    x_grad_ela = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    loss_ela = model_ela(x_grad_ela).sum()
    loss_ela.backward()
    assert model_ela.delta_net.q_proj.weight.grad is not None, "DeltaNet q_proj gradient missing"
    assert model_ela.expander.wexpand.weight.grad is not None, "Expander wexpand gradient missing"
    assert model_ela.out_proj.weight.grad is not None, "out_proj gradient missing"
    print("  [Gradient] OK")

    # 4. Irregular lengths (not a multiple of expand_ratio)
    seq_len2 = 63
    x2 = torch.randn(batch_size, seq_len2, d_model, device=device)
    out2 = model_ela(x2)
    assert out2.shape == (batch_size, seq_len2, d_model), f"Irregular shape: {out2.shape}"
    print(f"  [Irregular {seq_len2}] OK")

    # 5. Overlap mode
    model_ela_overlap = ExpandLinearAttention(
        d_model=d_model,
        num_heads=num_heads,
        expand_ratio=2,
        overlap=True,
        delta_net_kwargs=delta_kwargs,
    ).to(device)
    out_overlap = model_ela_overlap(x)
    assert out_overlap.shape == (batch_size, seq_len, d_model), f"Overlap shape: {out_overlap.shape}"
    assert torch.isfinite(out_overlap).all(), "Overlap output NaN/Inf"
    print("  [Overlap] OK")

    print("All tests passed for ExpandLinearAttention.")