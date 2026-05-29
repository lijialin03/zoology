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

class GatedPoolCompressor(nn.Module):
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


class GatedPoolExpander(nn.Module):
    """
    Expands sequence length by factor `expand_ratio` using gated linear projection.
    Designed as the inverse of GatedPoolCompressor.

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

        # Linear layers to produce raw expansion features and gating logits
        self.wexpand = nn.Linear(d_model, coff * expand_ratio * d_model, bias=False)
        self.wgate = nn.Linear(d_model, coff * expand_ratio * d_model, bias=False)
        # Learnable bias for each expanded position (shared across all tokens)
        self.ape = nn.Parameter(torch.empty(coff * expand_ratio, d_model))
        nn.init.normal_(self.ape, mean=0.0, std=0.02)

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

        # Project to raw expansion and gates
        expand = self.wexpand(x)                # [B, S, coff*ratio*D]
        gate = self.wgate(x)                    # [B, S, coff*ratio*D]

        # Reshape to separate the expansion dimension
        expand = expand.view(B, S, coff * ratio, D)
        gate = gate.view(B, S, coff * ratio, D)

        # Add learnable bias to gates
        gate = gate + self.ape.view(1, 1, -1, D)

        # Sigmoid to produce weights in (0,1)
        gate = torch.sigmoid(gate)

        # Gated expansion
        out = expand * gate                     # [B, S, coff*ratio, D]

        if self.overlap:
            out = self._overlap_combine(out)    # [B, S*ratio, D]
        else:
            # No overlap: simply flatten
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
        self.compressor = GatedPoolCompressor(
            d_model=d_model,
            head_dim=self.head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
        self.block_proj = nn.Linear(self.head_dim, d_model)
        self.delta_net = DeltaNet(d_model=self.head_dim, **delta_net_kwargs)
        self.out_proj = nn.Linear(d_model, d_model * compress_ratio)
        self.block_pos_emb = nn.Embedding(compress_ratio, d_model)

        self.res_proj = nn.Linear(d_model, d_model)
        self.res_norm = nn.RMSNorm(d_model)

    def forward(self, x):
        # x: [B, S, D]
        if self.use_sliding:
            x = self.sliding_attn(x) + x  # residual

        compressed = self.compressor(x) # [B, n_blocks, head_dim]
        # 投影回 d_model
        # compressed = self.block_proj(compressed) # [B, n_blocks, D]
        # DeltaNet 处理压缩序列
        dn = self.delta_net(compressed)         # [B, n_blocks, head_dim]

        dn = self.block_proj(dn)         # [B, n_blocks, D]
        # 将输出上采样回原始长度
        dn_expanded = dn.unsqueeze(2).expand(-1, -1, self.compress_ratio, -1)  # [B, n_blocks, ratio, D]
        pos_emb = self.block_pos_emb.weight.unsqueeze(0).unsqueeze(0)           # [1, 1, ratio, D]
        dn_expanded = dn_expanded + pos_emb
        out = dn_expanded.view(dn_expanded.size(0), -1, self.d_model)[:, :x.shape[1], :]

        # out = out + self.res_proj(x)
        # out = out + self.res_norm(x)
        return out

        
        # out = self.out_proj(dn)  # [B, n_blocks, D * ratio]
        # out = out.view(out.size(0), -1, self.d_model)  # [B, n_blocks * ratio, D]
        # return out[:, :x.shape[1], :]

        # # 压缩时
        # mean = kv.mean(dim=2)                # [B, n_blocks, head_dim]
        # residual = kv[:, :, 0, :] - mean     # 取第一个 token 与均值的差
        # # 将 mean 和 residual 拼接后投影到 d_model（或分别处理）
        # compressed = torch.cat([mean, residual], dim=-1)  # [B, n_blocks, 2*head_dim]
        # compressed = self.block_proj(compressed)          # -> [B, n_blocks, d_model]
        # # DeltaNet 处理...
        # # 上采样时，将 DeltaNet 输出再拆分为 mean' 和 residual'，然后重建
        # mean_out, residual_out = dn.chunk(2, dim=-1)
        # reconstructed = mean_out.unsqueeze(2) + residual_out.unsqueeze(2)  # 简单重建
        # reconstructed = torch.cat([reconstructed, ...], dim=2)   # 需要恢复 ratio 个 token

    # ------------------------------------------------------------------
    def state_size(self, sequence_length: int = 2048) -> int:
        """Compute total state size (for memory accounting).

        Returns:
            State size = delta_net.state_size
        """
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
                GatedPoolCompressor(
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

    # DeltaNet 参数（轻量设置）
    kwargs = {
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
        **kwargs,
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
    assert model.block_proj.weight.grad is not None, "block_proj weight gradient missing"
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