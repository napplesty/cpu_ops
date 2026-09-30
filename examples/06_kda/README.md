# 06 — KDA（Kimi Delta Attention，门控 delta 规则线性注意力）

Kimi Linear（Moonshot AI）的线性注意力组件：vector-wise 门控 delta 规则。
`kda.h` 提供**两种执行形态**（用户所需的功能核心）：

- **naive（递归）**：逐 token 状态机，每 token O(dk·dv)——decode 路径，
  状态经 `s0` / `s_out` 跨调用传递。
- **chunked（块式并行）**：块内把耦合的 delta 校正解耦为单位下三角
  前代 + 注意力形状的核，状态更新摊销到每**块**一次 O(dk·dv)——
  prefill / 训练路径。

**递归（规范定义；每 head，状态 S ∈ R^{dk×dv}）**

```
S ← λ_t ⊙ S                  # 逐通道衰减（KDA 相对标量衰减的表达力来源）
δ_t = v_t − k_tᵀ S           # delta：残差 = v 减去当前状态的预测
S ← S + β_t · k_t δ_tᵀ       # β_t = σ(标量 logit)
o_t = (scale·q_t)ᵀ S         # 更新后读出
```

**块内展开**（块长 C，Λ_t = λ_1⊙…⊙λ_t，G = logΛ 累积；C=2 手工验证）：

```
Δ = (I+N)⁻¹ (V − K̃ S₀)          N[t][s] = β_s·⟨k_t, k_s⟩_decay   （严格下三角，前代求解）
o_t = (q_t⊙Λ_t)ᵀS₀ + Σ_{s≤t} A[t][s]·β_sδ_s
S_new = diag(e^{G_C})S₀ + Σ_t (k_t⊙e^{G_C−G_t})⊗β_tδ_t
```

**数值设计**：所有出现的衰减指数都是相对累积对数衰减 `G_t − G_s ≤ 0`
（log λ ≤ 0），逐对计算 exp 永不溢出、下限溢出即数学正确极限——因此
**精确**，不需要 FLA 风格的因子分解 + 钳位近似。O(C²) 的成对核在块衰减
范围 `max|G_last| ≤ 140` 时改用**中点分裂**精确因子化
`⟨a_t,b_s⟩_decay = (a⊙e^{G_t−m})·(b⊙e^{m−G_s})`（m = G_last/2，两侧因子
都在 e^±70 内，exp 移出成对循环）；更极端的块回退逐对相对指数核（对
任意衰减精确）。分支只依赖数据，逐位稳定性不受影响。k 按上游给定消费
（Kimi Linear 在上游做 L2 归一化，使 N 收缩、前代良态）；`logσ` 用
`−softplus(−x)` 稳定式。

**确定性**：每个 (batch, head) 任务内固定顺序（前代按 t 升序、SIMD 车道
固定归约树），任意线程数**逐位一致**；两种模式彼此只保证到浮点舍入
（求和顺序不同）——自检含直接互证。

**自校验**：双精度 naive 递归即参考（naive 版本身就是规范）；两模式各自
对照；f32 下 chunked ≡ naive 到 2e-3；**decode 状态往返契约**——块网格对
齐时两次 sq=8 调用（经 s0/s_out 传递状态）与一次 sq=16 调用 f32 **逐位
一致**（中间状态精确往返）；1T vs 4T 与跨运行 memcmp；错误路径。

**Build & run**

```sh
cmake -B build && cmake --build build -j4 --target kda
./build/examples/06_kda/kda           # 自校验
./build/examples/06_kda/kda --bench   # naive vs chunked
```

**Benchmark**（Apple M1 Pro，4T，B=1 H=32，dk=dv=128，C=64，bf16，
best-of-2/3）

| seq | naive 4T ms | chunk 1T ms | chunk 4T ms | chunk/naive |
|---|---|---|---|---|
| 2048 | 113.4 | 281.0 | 78.5 | 1.44 |
| 8192 | 446.1 | 1119.7 | 306.5 | 1.46 |

块内的六个矩阵运算（A/N 核 `QQ·K̂ᵀ`/`KK·K̂ᵀ`、`rhs = −K̃S+V`、`Q̃S`、
`A·BD`、`S += K̂ᵀ·BD` 原地累加）在因子化安全的块上走库 GEMM 原语（单
线程逐 (batch,head) 任务调用，与 MLA 吸收 GEMM 同模式），衰减组合行
（k̃/q̃/k̂）与因子化行用 `exp_scale_row` 的向量 exp2 多项式构建（替代每
块 ~6·C·dk 次标量 expf）；前代求解与极端衰减回退（|G|>140）保留标量
路径。GEMM 化后 chunked 反超 naive ~1.45×（此前行核版为 0.77×）；naive
路径仍是 decode 内核（每 token O(dk·dv) 常数工作 + 状态跨调用传递），
长序列 prefill 用 chunked。

**与 02-05 示例的关系**：这是第一个非 softmax 注意力（无 running max、
无归一化分母），权重变换从 `exp2(x−m)` 换成 sigmoid 门控衰减——即
`attention/attention.h` 里预留的第三条扩展缝。复用 04 的流式行核原语。
