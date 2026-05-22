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

from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

from zoology.mixers.csa_hca import GatedPoolCompressor
from zoology.mixers.delta_net import DeltaNet
from zoology.mixers.slide_attn import SlidingAttn


class CompressedLinearAttention(nn.Module):
    def __init__(
            self, 
            d_model: int,
            num_heads: int,
            use_sliding: bool = False,
            compress_ratio: int = 4,
            window_size: int = 32,
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
            overlap=True
        )
        self.block_proj = nn.Linear(self.head_dim, d_model)
        self.delta_net = DeltaNet(d_model=d_model, **kwargs)
        self.out_proj = nn.Linear(d_model, d_model * compress_ratio)
        self.block_pos_emb = nn.Embedding(compress_ratio, d_model)

    def forward(self, x):
        # x: [B, S, D]
        if self.use_sliding:
            x = self.sliding_attn(x) + x  # residual
        compressed = self.compressor(x) # [B, n_blocks, head_dim]
        # 投影回 d_model
        compressed = self.block_proj(compressed) # [B, n_blocks, D]
        # DeltaNet 处理压缩序列
        dn = self.delta_net(compressed)         # [B, n_blocks, D]
        # 将输出上采样回原始长度
        
        dn_expanded = dn.unsqueeze(2).expand(-1, -1, self.compress_ratio, -1)  # [B, n_blocks, ratio, D]
        pos_emb = self.block_pos_emb.weight.unsqueeze(0).unsqueeze(0)           # [1, 1, ratio, D]
        dn_expanded = dn_expanded + pos_emb
        out = dn_expanded.view(dn_expanded.size(0), -1, self.d_model)
        return out[:, :x.shape[1], :]

        
        # out = self.out_proj(dn)  # [B, n_blocks, D * ratio]
        # out = out.view(out.size(0), -1, self.d_model)  # [B, n_blocks * ratio, D]
        # return out[:, :x.shape[1], :]

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