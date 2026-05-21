# CSA 与 HCA 设计分析

`model.py` 是一个生产级 DeepSeek-V4 模型，其 Attention 层根据 `compress_ratios` 为不同层分配不同的压缩策略：

```python
compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
# n_layers=7: 层0=纯滑动窗口, 层1=纯滑动窗口, 层2=CSA, 层3=HCA, 层4=CSA, 层5=HCA, 层6=CSA
```

相比之下：
- **`deepseek_nsa.py`**：每层同时运行三条通路（compressed + fine/selected + sliding window），通过 learned gating 融合
- **`attention.py`**：标准 MHA，纯 dense attention 基线

---

## 一、CSA（Compressed Sparse Attention）— `compress_ratio=4`

CSA 的核心思路是：**先压缩再稀疏选择**，用极少的压缩 token 替代全量 KV cache。

### 1.1 压缩模块 — `Compressor`（model.py:279-377）

与 NSA 用 MLP 压缩不同，CSA 使用**门控池化（Gated Pooling）**：

```
输入: [B, S, D] 的 KV 序列
  ↓ wkv (Linear)    ↓ wgate (Linear)
  kv [B,S,d]        score [B,S,d]
  ↓                 ↓ + ape (可学习位置偏置)
  ↓                 ↓ softmax over compress_ratio tokens
  ↓                 ↓
  kv * softmax(score) → sum → 1个压缩token [B,1,d]
```

关键设计细节：
- **Overlapping windows**（ratio=4 时启用）：相邻压缩窗口重叠一个 token，让压缩边界更平滑（`overlap_transform` 方法，第307-314行）
- **增量 decode 支持**：通过 `kv_state` / `score_state` buffer 维护未完成的压缩窗口（第303-304行）
- **位置编码**：压缩后的 token 应用 RoPE（第367行），保持时序信息
- **可选 Hadamard 旋转**：Indexer 内部的 Compressor 使用 `rotate=True`，压缩后做 Hadamard transform + FP4 量化（第368-370行），用于低精度索引

### 1.2 稀疏选择 — `Indexer`（model.py:380-433）

Indexer 只在 **CSA 层（ratio=4）** 启用，负责从压缩后的 KV 中选出 top-k 个重要位置：

```
Q (query) → wq_b → apply RoPE → Hadamard旋转 → FP4量化
                                               ↓
X → Compressor → 压缩KV cache ──────────────→ Q·K^T → relu → × weights → topk
```

与 NSA 的 selection 对比：

| 特性 | CSA Indexer | NSA Selection |
|------|------------|---------------|
| 压缩方式 | Gated Pooling + Hadamard | MLP (Linear→ReLU→Linear) |
| 重要性分数 | 独立 Q·K 计算 + relu | 复用 coarse attention 的 softmax |
| 量化 | FP4 QAT 模拟 | 无 |
| Head 结构 | 独立 `index_n_heads=64` | 与 attention head 共享 |

### 1.3 稀疏注意力 — `sparse_attn` kernel（model.py:528,533）

CSA 层的最终 attention 将三种 KV 来源拼接：

```
topk_idxs = [window_tokens] + [indexer_selected_compress_tokens]

sparse_attn(q, kv=[window_kv + compressed_kv], attn_sink, topk_idxs, scale)
```

- 有 **attention sink**（`attn_sink`，第456行）：每个 head 一个可学习标量，吸收无关 token 的注意力
- 通过 `topk_idxs` 指定每个 query token 关注的具体 KV 位置，跳过不重要的区域

---

## 二、HCA（Heavily Compressed Attention）— `compress_ratio=128`

HCA 的设计理念是：**极限压缩 + 不做稀疏选择**，用极少的内存覆盖全部历史。

### 2.1 与 CSA 的关键区别

```python
# model.py:466-471
if self.compress_ratio:
    self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
    if self.compress_ratio == 4:
        self.indexer = Indexer(args, self.compress_ratio)  # ← 仅 CSA 有
    else:
        self.indexer = None  # ← HCA 没有 Indexer
```

HCA 层的 attention 只有两种 KV 来源：

```
topk_idxs = [window_tokens] + [all_compress_tokens]  # 全部压缩token，不筛选

# compress_topk_idxs 直接用 get_compress_topk_idxs 取所有块（第513行）
```

### 2.2 HCA Compressor 的特点

- **无 overlap**：ratio=128 时 `overlap=False`，不做重叠窗口
- **无 Hadamard 旋转**：`rotate=False`，不做额外变换
- **压缩比 128x**：128 个连续 token 压缩成 1 个，极大减少 KV cache
- **无稀疏选择**：所有压缩块都参与 attention，不做 topk 筛选（因为总共也没几个块）

### 2.3 设计动机

HCA 替代了 NSA 中的 "fine/selected attention" 通路。当压缩比足够大（128x）时：
- 压缩块总数已经很少（4096/128 = 32 个块），不需要再筛选
- 直接用 dense attention 覆盖全部压缩历史，计算量可控
- 配合滑动窗口（128 tokens）保证局部精度

---

## 三、三层对比总结

```
                     MHA基线          NSA                    CSA (ratio=4)          HCA (ratio=128)
                     (attention.py)   (deepseek_nsa.py)      (model.py)             (model.py)
──────────────────────────────────────────────────────────────────────────────────────────────
压缩方式             无               MLP压缩                门控池化(Gated Pool)    门控池化(无overlap)
稀疏选择             无               coarse attn→topk       独立Indexer(Q·K+relu)  无(全部压缩块)
滑动窗口             无               ✓(LocalAttention)      ✓(window_size=128)     ✓(window_size=128)
多通路融合           无               3路learned gating      单路sparse_attn        单路sparse_attn
每层策略             统一             统一(3路并行)          按层分配不同ratio       按层分配不同ratio
Attention Sink       无               无                     ✓(可学习)              ✓(可学习)
量化                 无               无                     FP4/FP8 QAT            FP8 QAT
```

核心设计思想：
1. **CSA** 通过适度压缩(4x) + 智能选择(Indexer topk)平衡效率与精度，适合中等长度上下文
2. **HCA** 通过极限压缩(128x)将所有历史压缩为常数级 token，放弃稀疏选择换取极致效率，适合超长上下文
3. 两者**按层交替部署**（层2=CSA, 层3=HCA, 层4=CSA...），浅层和深层用纯滑动窗口，形成多尺度注意力金字塔