# 05 — CSA（Compressed Sparse Attention，NSA 式三分支稀疏注意力）

DeepSeek Native Sparse Attention（arXiv 2502.11089）的三分支设计，组装在
MLA 潜空间缓存之上。`csa.h` 复用 `04_dsa` 的流式行核原语
（`dot_f32` / `axpy_f32`），不链接进 `libcpu_ops.a`。

**算法**（每个 query 三路并行、sigmoid 门控加权求和）：

```
o_t = g_cmp·Attn(q̃, 块质心) + g_slc·Attn(q̃, 选中块的原始 token) + g_win·Attn(q̃, 最近 w 个 token)
```

1. **压缩分支**：每块一行粗粒度表示。论文的 φ 是块内可学习 MLP
   （l=32/d=16 重叠块）；本例取对齐退化情形 l = d = l′ = 单一 `block`
   参数，映射二选一——外部注入 `kv_compressed [b][nblk][dc+dp]`
   （部署态：φ 在上游算好随 cache 维护），或 null 时代为块均值
   （记为简化）。
2. **选块分支**：块重要性直接**复用压缩分支的 softmax 后分数**，跨
   attention head 求和（论文的 GQA/MQA 规则）→ 每个 query 行一套
   选集、全部 head 共享。取 top-`select_blocks` 个 64-token 块**整块
   gather**、块内按原始 token 粒度做注意力；论文的固定激活（首块 +
   2 个最近局部块）计入预算。
3. **窗口分支**：最近 `window` 个 token，连续区间，无需 gather。

**因果边界规则**（论文未明确，本例补全并记录）：压缩/选块分支只看
**完整块**（块 l 对位置 p 的 query 合法 ⇔ (l+1)·block ≤ p+1）；包含 p
的不完整块交给窗口分支。非因果模式全部块合法（含尾部残块），窗口取
序列末尾。

**执行结构**：每 batch 先算一次质心（f32，跨行跨 head 共享）；行按
内存预算分块处理——phase A（行×head）：q̃ 吸收 + 压缩分支一遍在线
softmax（顺手得到每 head 的 softmax 块分数）；phase A2（行）：头求和
→ 选块（含固定块）→ 整块 gather 一次共享；phase B（行×head）：选块
+ 窗口两路行核 + 门控 + 反吸收。全部走无 pack 流式行核，q̃ 全程 f32。

**确定性**：质心、softmax、头求和、选块（分数降序/块号升序全序 +
固定规则）、分支注意力均按固定顺序归约——任意线程数结果**逐位一致**
（自检含 1T vs 4T 与重复运行 memcmp 验证）。

自校验参考实现用双精度重算全流水线；选块规范锚点是算子的 f32 块分数
镜像（`block_scores_out`，头求和后的 softmax 分数），镜像本身另行对照
双精度头求和校验接线——与 DSA 示例同一模式。

**Build & run**

```sh
cmake -B build && cmake --build build -j4 --target csa
./build/examples/05_csa/csa           # 自校验
./build/examples/05_csa/csa --bench   # 参考基准（vs 稠密 MLA）
```

**Benchmark**（Apple M1 Pro，4T，B=1 H=32，dn/dp/dc=128/64/512，
block=64、top-16 块（1024 token）、window=512，bf16，best-of-5）

decode（seq_q=1，causal，对同形稠密 MLA）：

| kv len | MLA 4T ms | CSA 1T ms | CSA 4T ms | 加速比 |
|---|---|---|---|---|
| 8192 | 32.8 | 13.8 | 6.0 | 5.5x |
| 32768 | 115.2 | 18.3 | 10.2 | 11.3x |
| 65536 | 225.7 | 23.9 | 16.0 | 14.2x |

prefill（chunked continuation 512×8192）：MLA 4T 1195 ms，CSA 4T 2259 ms
（0.53×，逐行选集的 M=1 行核代价，与 DSA 示例同一已知限制）。

decode 随 skv 增长很缓（6→10→16 ms）：打分直接复用压缩分支（64k 下
仅 1024 行质心点积/头），选中+窗口两路触达固定为 1024+512 行——这是
对 DSA（独立 64×128 indexer，537M MAC）的结构性优势，代价是块粒度
选择的近似。

**与 DSA 示例的差异**：DSA 逐 token 精确 top-k（token 粒度选择），
CSA 逐块 top-k 且多出压缩/窗口两路——CSA 的注意力触达是
`nblk + n·g + w` 行（64k 下 ≈ 1024+1024+512），比 DSA 的选集更小；
代价是压缩分支的粗粒度近似。两者共享同一套无 pack 行核原语。
