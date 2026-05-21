# DeltaNet / CLA / CSA 对比分析

---

## 问题1：为什么 CLA 要增加 Sliding Window Attention？

### 原始 DeltaNet 的架构

原始 DeltaNet 是**纯线性注意力**（delta rule），没有 sliding window attention。其核心机制是：

- 维护一个固定大小的记忆矩阵：`S_t = S_{t-1} + v_t ⊗ k_t^T`（O(d²) 状态）
- 每个新 token 通过 delta rule 增量更新该矩阵
- 输出 = `S_t · q_t`

### CLA 增加 Sliding Window 的原因

**核心设计哲学**：CSA 的 MQAR 实验已经验证，多路径设计（局部 + 压缩 + 选择）远优于单一机制：
- CSA 达 96.48%（d256）
- NSA 仅 66.62%（d256）

**Sliding Window 的不可替代性**：

1. **捕获细粒度局部模式**：相邻 token 之间的精确关系（如 MQAR 中连续 key-value 对）
2. **线性注意力的信息丢失**：DeltaNet 通过 `S = Σ v_i k_i^T` 压缩所有历史信息到一个固定 d×d 矩阵中，这是一个有损压缩过程，局部精确信息在压缩时会被"抹平"
3. **类比理解**：
   - Sliding window = "显微镜"（局部精确，范围小）
   - DeltaNet 压缩注意力 = "望远镜"（全局概览，细节模糊）
   - 两者**互补**而非替代

**为什么不能只用 DeltaNet**：

DeltaNet 的 O(d²) 状态容量是固定的，与序列长度无关。对于需要精确局部匹配的任务（如 MQAR 中相邻位置的 key-value 检索），这个固定容量的压缩表示可能丢失关键的局部信息。Sliding window 提供了一条"无损"的局部信息通道。

---

## 问题2：显存/KV Cache 对比分析

以 `d_model=128, num_heads=2, window_size=16, compress_ratio=4, index_topk=4` 为例。

### 2.1 DeltaNet（纯线性注意力）

```
状态 = num_heads × head_k_dim × head_v_dim
     = 2 × 64 × 64
     = 8,192 个元素
     = 32 KB (float32)
```

| seq_len | KV Cache | 增长率 |
|---------|----------|--------|
| 256     | 32 KB    | -      |
| 1024    | 32 KB    | 0      |
| 2048    | 32 KB    | 0      |
| 8192    | 32 KB    | 0      |

**特点**：KV Cache 与序列长度**完全无关**（O(1)），因为 DeltaNet 将序列压缩为一个固定大小的 d×d 矩阵。

### 2.2 CLA（Compressed Linear Attention）

```
状态 = sliding_window KV          + DeltaNet 状态
     = window_size × d_model × 2  + num_heads × head_k_dim × head_v_dim
     = 16 × 128 × 2               + 2 × 64 × 64
     = 4,096                      + 8,192
     = 12,288 个元素
     = 48 KB (float32)
```

| seq_len | KV Cache | 增长率 |
|---------|----------|--------|
| 256     | 48 KB    | -      |
| 1024    | 48 KB    | 0      |
| 2048    | 48 KB    | 0      |
| 8192    | 48 KB    | 0      |

**特点**：KV Cache **也与序列长度无关**（O(1)）。Sliding window 受 window_size 定界，不随 seq_len 增长。

### 2.3 CSA（Compressed Sparse Attention）

```
状态 = (window_size + n_comp + index_topk) × d_model × 2
     = (16 + n_comp + 4) × 128 × 2
```

| seq_len | n_comp | 状态（元素） | KV Cache（float32） | 相对 DeltaNet |
|---------|--------|-------------|---------------------|---------------|
| 256     | 64     | 21,504      | 84 KB               | 2.6×          |
| 512     | 128    | 37,888      | 148 KB              | 4.6×          |
| 1024    | 256    | 70,656      | 276 KB              | 8.6×          |
| 2048    | 512    | 136,192     | 532 KB              | 16.6×         |
| 4096    | 1024   | 267,264     | 1.04 MB             | 33.4×         |
| 8192    | 2048   | 529,408     | **2.07 MB**         | 66.2×         |

**特点**：KV Cache **随序列长度线性增长**（O(n)）。n_comp = seq_len / compress_ratio，所以路径 2（压缩注意力）的 KV cache 仍然正比于 seq_len。

### 2.4 汇总对比

```
KV Cache 大小 vs 序列长度 (d_model=128)

 2.0 MB ─┤                                    ● CSA
         │                                  ╱
 1.5 MB ─┤                              ╱
         │                           ╱
 1.0 MB ─┤                       ╱
         │                    ╱
 500 KB ─┤                ╱
         │            ╱╱
 100 KB ─┤––––––––––––––––––––––––––––––––  ─ CLA (48 KB)
  32 KB ─══════════════════════════════════  ─ DeltaNet (32 KB)
         │
         └─────┬─────┬─────┬─────┬─────┬─────
              256   512   1024  2048  4096  8192  seq_len
```

### 2.5 综合对比表

| 指标 | DeltaNet | CLA | CSA |
|------|----------|-----|-----|
| KV Cache 增长 | **O(1)** | **O(1)** | O(n) |
| 局部精确建模 | 弱 | **强** (sliding window) | **强** (sliding window) |
| 长程依赖 | **强** (d² state) | **强** (d² state) | 中 (compressed attn) |
| 稀疏选择 | 无 | **有** (Indexer top-k) | **有** (Indexer top-k) |
| seq_len=8192 时 KV Cache | 32 KB | 48 KB | 2.07 MB |
| seq_len=8192 时相对 DeltaNet | 1× | 1.5× | **66.2×** |
| 推理耗时 | O(d²) per token | O(d² + w·d) per token | O(n·d) per token |

---

## 问题3：为什么 CLA 的 MQAR 准确率不随 num_kv_pairs 变化？

### 3.1 实验现象

MQAR 实验（d_model ∈ {64, 128, 256} × 4 LR × 32 epochs, random_non_queries=False），heatmap 和 per-slice 数据揭示了 CLA 的"二元"行为：

| 架构 | d_model | KV=4 | KV=8 | KV=16 | KV=32 | KV=64 | KV=128 | KV=256 |
|------|---------|------|------|-------|-------|-------|--------|--------|
| **CLA** | 64 | 0.47 | 0.48 | 0.47 | 0.47 | 0.47 | 0.47 | 0.47 |
| **CLA** | 128 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| **CLA** | 256 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| **DeltaNet** | 64 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.56 | 0.19 |
| **DeltaNet** | 128 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.76 |
| **DeltaNet** | 256 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 |

- **CLA**：d128/256 时所有 KV pair 全部 ~0.99（天花板）；d64 时全部 ~0.47（地板）。**无渐进退化**。
- **DeltaNet**：小 KV 对准确率完美，大 KV 对逐渐下降。**典型的容量饱和曲线**。

### 3.2 根因：Indexer 将检索复杂度与 KV 对数解耦

CLA 有三条路径，但在 MQAR 任务中起决定性作用的是 **Pathway 3（Indexer + DeltaNet）**：

```
原始序列 (seq_len=1024, 256对KV)
  → compress_ratio=4 avg-pool → 64 个压缩块
  → Indexer 选 top-4 个块
  → DeltaNet 处理 4 个块（含 8 对 KV）
  → 加权输出
```

**Pathway 1（Sliding Window, window=16）**：KV pair 在序列开头，query 在末尾，窗口**永远够不到**。

**Pathway 2（Compressed DeltaNet）**：64 个压缩块远超 DeltaNet 状态矩阵的秩（d128 时 32×32=1024 容量 ≈ 32 个独立关联），无法可靠存储。

**Pathway 3（Indexer top-k=4）**：Indexer 只需从 64 个压缩块中**选出正确的 4 个**，每个块含 2 对 KV。DeltaNet 仅需处理 8 对 KV → 远低于状态容量。

**核心洞察**：Indexer 将 MQAR 从 "存储 N 对 KV 关联" 降维为 "在 64 个候选中找到正确的 4 个块"。复杂度与 num_kv_pairs **解耦** — 无论 4 对 KV 还是 256 对 KV，Indexer 都只选 4 个块。

### 3.3 d_model=64 时 Indexer 为何失效

Indexer 的核心操作是 **head_dim 维空间中的点积匹配**：

```
idx_scores = Q_index @ K_compressed^T / sqrt(head_dim)
```

在 d_model=64, num_heads=4 时，head_dim=16。问题链条：

1. **嵌入空间极度拥挤**：vocab_size=8192 的 token 嵌入到 16 维 → 大量 token 的嵌入向量高度相似
2. **avg-pool 压缩进一步模糊信息**：K_compressed 是 4 个连续 token 投影的均值（跨越 KV 对边界），判别信息被稀释
3. **16 维点积的判别力不足**：Indexer 无法可靠区分"包含目标 KV 对的压缩块"和"无关块"
4. **Indexer 选错块 → Pathway 3 失效** → CLA 退化到 ~0.47（非完全随机，而是部分碰撞匹配的结果）

**对比 DeltaNet d64**：DeltaNet d64 能完美解决 KV≤32（acc=1.0），因为它**不做压缩**，直接对原始 token 做 DeltaNet。16 维虽小，但对原始 token（未经 avg-pool 稀释）已足够区分。

### 3.4 DeltaNet vs CLA 的退化模式对比

| 维度 | DeltaNet | CLA |
|------|----------|-----|
| 处理方式 | 所有 token 直通 DeltaNet | avg-pool 压缩 → Indexer 选 top-4 → DeltaNet |
| 状态存储负担 | 需存储所有 KV 对（随 num_kv_pairs 线性增长） | 仅需存储 8 对 KV（恒定，与 num_kv_pairs 无关） |
| 瓶颈 | **状态矩阵容量 O(d²)**：KV 对数 > rank(S) 时退化 | **Indexer 判别力**：head_dim 决定能否区分正确块 |
| 退化模式 | **渐进式**：容量逐步饱和 → 准确率逐步下降 | **二元式**：Indexer 能区分 → ~1.0；不能 → 退化到碰撞匹配水平 |
| 对 d_model 的敏感度 | 低（d=64 即可解决大部分 KV 对） | 高（d=64 Indexer 完全失效，d=128 完美） |

### 3.5 改进方向

如果想让 CLA 在 d64 上也展现合理的趋势曲线，可以尝试：

| 参数 | 当前值 | 改进方向 | 效果 |
|------|--------|----------|------|
| `index_topk` | 4 | 增大到 8–16 | 扩大检索范围，降低对 Indexer 精确度的要求 |
| `compress_ratio` | 4 | 减小到 2 | 减少 avg-pool 信息损失，每个压缩块只含 1 对 KV |
| `expand_k` | 1.0 | 增大到 2.0 | head_k_dim 从 16 → 32，Indexer 判别力翻倍 |
| `num_heads` | 4 | 增大到 8 | 多头提供冗余，部分 head 的 Indexer 可能成功 |

---

## 结论

**CLA 的核心价值**：继承了 CSA 验证过的三路径架构（局部 + 压缩 + 选择），但将路径 2 和 3 的 softmax 注意力替换为 DeltaNet 线性注意力，使得：

1. **KV Cache 从 O(n) 降为 O(1)**：对于长序列场景（如 8K+ tokens），CLA 的显存占用仅 48 KB，而 CSA 达到 2+ MB
2. **保留了局部建模能力**：Sliding window 确保局部精确信息不丢失
3. **保留了稀疏选择机制**：Indexer 的 top-k 选择在 DeltaNet 的压缩输出上仍然有效
4. **推理效率更高**：每 token 推理从 O(n·d) 降为 O(d² + w·d)，其中 w 为常数 window_size

**适用场景**：
- **DeltaNet**：长序列建模，对局部精度要求不高的任务
- **CLA**：长序列 + 需要局部精确匹配的任务（如 MQAR、长文档 QA）
- **CSA**：中短序列，追求最高精度的任务（显存可承受时）