# 04 — DSA（DeepSeek Sparse Attention，DeepSeek-V3.2）

在 MLA 潜空间缓存之上组装 DSA：lightning indexer 打分 → 每 query 精确 top-k
选 token → CLS 汇总 token → 无 pack 的稀疏行核。`dsa.h` 只依赖库原语
（`simd::Vec`、线程池），不链接进 `libcpu_ops.a`——CUTLASS 式的组装示例。

**算法**（按 DeepSeek-V3.2 论文公式 + cuDNN DSA API 语义实现）

1. **Lightning indexer**（MQA 结构，V3.2 为 64 头 × 128 维、单共享 key）：

   ```
   score(t, s) = Σ_j w[t,j] · ReLU(indexer_scale · q_idx[t,j] · k_idx[s])
   ```

   分数是每 (query, token) 一个标量（跨 indexer 头求和），因此**所有
   attention head 共享同一个选集**。`k_idx` 是逐 token 预计算的输入缓存
   （与 MLA 潜缓存并列存放，V3.2 里为 FP8）。
2. **选 token**：因果前缀上的精确 top-k（V3.2 取 k=2048），同分取小
   index，选中集合升序。前缀不足 k 时取全部（vLLM 语义）。
3. **CLS token**：对整条因果前缀做 `softmax(score)` 加权求和得到一个
   query 专属的汇总 KV 行（潜空间 ‖ rope），拼接在 k 个选中行之后作为
   第 k+1 行——未选中 token 的“单 token 摘要”。论文正文未给出 CLS
   定义，此处按官方推理代码的语义实现（`softmax(indexer 分数)` 全前缀
   加权和，K/V 同权），并留了 `use_cls` 开关做消融。
4. **稀疏注意力**：MLA 吸收式——每 head `q̃ = [W_UK·q_nope ‖ q_pe]`，
   对 gather 出来的 (k+1)×576 行做在线 softmax，V 取潜前缀 512 维，
   输出 `o = W_UV·õ`。

**CPU 取舍**（刻意不照搬 GPU kernel）：

- **精确 top-k 而非两级粗筛**：64k 条 f32 分数行只有 256 KiB，驻 L2，
  `std::partial_sort` 确定且精确；GPU 的 block-max/radix-select 是为
  不物化稠密分数设计的，CPU 上没有收益。
- **gather 成稠密再算**：每个 query 行的选中行一次性拷进连续缓冲，
  全部 head 共享，而不是在每个 head 的 kernel 里重复 gather。
- **无 pack 的 M=1 行核**：每个 query 行有自己的选集，M=1/（行,head），
  GEMM 核的 panel 打包会把同一份 gather 面板按 head 重复打包。行核
  顺流一次读尽、q̃ 全程 f32，这也是路线图上“decode 专用小 M/无 pack
  kernel”的第一个实例。
- **indexer 双路径**：行块 ≥ 8 时按 indexer 头走库的 GEMM 原语
  （`[rows×di]·[di×skv]`，点积缩放放进 epilogue），decode（rows=1）
  保留无 pack 的流式 GEMV——M=1 走 GEMM 会在每个 indexer 头上重打包
  整条 key 缓存。块大小只由问题形状决定，两条路径都逐位稳定。
- **indexer 数值跑 f32**：V3.2 用 FP8 是省 HBM 带宽；CPU 上的对应物是
  int8 点积 GEMM 原语（待做，见主 README 路线图）。

**确定性**：所有归约顺序与线程数无关（indexer 点积按 s 升序、CLS
按固定 8 片序合并、在线 softmax 按 gather 行序），任意线程数下结果
逐位一致；自检含 1T vs 4T 与重复运行的 memcmp 验证。

自校验参考实现用双精度重算整条流水线，唯一规范锚点是算子的 f32
indexer 分数镜像（`indexer_scores_out`）：选集与 CLS 权重定义在这份
f32 分数上（top-k 边界是部署态 f32 分数的性质），镜像本身另行对照
双精度 indexer 公式校验接线。

**Build & run**

```sh
cmake -B build && cmake --build build -j4 --target dsa
./build/examples/04_dsa/dsa           # 自校验
./build/examples/04_dsa/dsa --bench   # decode/prefill 对比稠密 MLA
```

**Benchmark**（Apple M1 Pro，4T，B=1 H=32，dn=128 dp=64 dc=512，
indexer 64×128，topk=2048，bf16 缓存，best-of-5）

decode（seq_q=1，causal，对同形稠密 MLA）：

| kv len | MLA 4T ms | DSA 1T ms | DSA 4T ms | 加速比 | 有效 GB/s |
|---|---|---|---|---|---|
| 8192 | 31.2 | 26.5 | 8.8 | 3.6x | 17.6 |
| 32768 | 115.4 | 53.7 | 17.1 | 6.7x | 34.2 |
| 65536 | 227.6 | 91.1 | 27.5 | 8.3x | 42.1 |

prefill（chunked continuation 512×8192）：MLA 4T 1132 ms，DSA 4T 3344 ms
（0.34×）。decode 是 DSA 的主战场——每 token 触达从整条 cache 降到
indexer key 流（64×128×2B/token）+ 一次 CLS 扫描 + (k+1) 行 gather，全部
无 pack 顺流，64k 上下文 8.3×（indexer 按 8 头分块后 k_idx 只扫 8 遍，
窄存→f32 走 `simd::widen_f16/widen_bf16` 向量宽化）。prefill 慢于稠密是当前实现的真实写照：
逐行精确选集使注意力只能以 M=1 行核算力执行（选中率 26% 省下的 FLOP
补不回行核与 GEMM 核的算力差），且窄存→f32 是逐元素软件转换——面板共享
选集重回 GEMM 核、NEON `fcvtl`/AVX2 向量加载原语都在路线图上（见主
README「已知限制」）。
