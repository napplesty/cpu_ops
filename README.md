# cpu-ops

CPU 高性能 GEMM 模板库：SIMD 微内核 + cache 分块 + 多线程。C++17，无第三方依赖，
纯头文件（header-only）CUTLASS 式分层模板库：包含即用，kernel 在使用方的编译单元内实例化。

- f32 / f64 GEMM：FMA 主路径
- int8 量化 GEMM（u8×s8→s32）：AVX-VNNI 点积指令；ARM dotprod/i8mm；
  AMX-INT8 瓦片 policy（实验性，opt-in）
- f16 / bf16 输入 GEMM（f32 累加 / 输出）：打包时转 f32；AVX512-BF16 上 bf16
  走 `vdpbf16ps` 原生点积（默认自动切换）
- MX 格式 GEMM（mxfp8 e4m3/e5m2、mxfp4 e2m1 + E8M0 块缩放）：解码后走 f32 FMA
  （AVX512F 上解码 16 路宽）；另有 opt-in 的 VNNI int8 点积路径（e2m1 精确）
- 组装式算子开发（CUTLASS 式）：fused attention（GQA/MHA/MQA）、MLA
  （DeepSeek 吸收式）、DSA（DeepSeek-V3.2 稀疏注意力）、CSA（NSA 式
  压缩稀疏注意力）与 KDA（Kimi Delta Attention，naive/chunk 双模式）
  作为 `examples/` 的组装参考，不进核心库——见「组织方式」
- ops 层原语：激活（gelu/silu）、归一化（rmsnorm / fused add+rmsnorm /
  layernorm）、精确 top-k（radix-select，与 partial_sort 逐位同语义）；
  融合 epilogue 目录含 GeLU/SiLu
- Split-K：深 k 自动切分 + 确定性两阶段归约
- 融合 epilogue：alpha/beta 之后按位置融合的 op 链（ReLU / Clamp / BiasAdd / …）
- ARM 支持：NEON / SVE2 定长 / SME（实验性）；int8 在 dotprod/i8mm 上自动启用
- 任意平台均有标量回退；单线程与多线程结果**逐位一致**

## 构建

```bash
cmake -B build && cmake --build build -j
ctest --test-dir build          # 23 个测试：13 个核心（simd/gemm/int8/f16/mx/fusion/ops + 标量回退）
                                 # + 10 个组装示例自校验（attention/mla/dsa/csa/kda × 普通/标量）
./build/basic_gemm              # 最小示例
./build/benchmark               # GFLOPS / GOPS 基准
./build/examples/02_fused_attention/fused_attention --bench   # fused attention
./build/examples/03_mla/mla --bench                           # MLA
./build/examples/04_dsa/dsa --bench                           # DSA（对比稠密 MLA）
./build/examples/05_csa/csa --bench                           # CSA（对比稠密 MLA）
./build/examples/06_kda/kda --bench                           # KDA（naive vs chunked）
```

## 组织方式（CUTLASS 式）

库只提供**原语与 GEMM 核心**；由原语**组装**出来的算子全部放在 `examples/`，
每个目录自包含：自己的头文件、自校验（double 参考 + 确定性断言）与基准合一的
main、目录级 `CMakeLists.txt` 与 README，并挂到 CTest（含标量回退构建）。这与
CUTLASS 的做法一致——其 FMHA 同样位于 `examples/41_fused_multi_head_attention`
而非核心库。要实现自己的算子，照抄一个目录：包含
`cpu_ops/arch/*`、`cpu_ops/gemm/threadblock/*` 原语头或相邻例子的头，在自己的编译单元里实例化。

- `examples/02_fused_attention/`：fused 块内核 + `Attention<T>` 入口
  （GQA/MHA/MQA，online softmax，causal 块跳过，decode KV-split；内核支持
  Q/K 归约深度与 V/O 宽度解耦 `dim_v`，供 MLA/稀疏变体复用）。
- `examples/03_mla/`：吸收式 MLA = 每 head 吸收 GEMM → fused 核心
  （`heads_kv=1`、共享 [c‖k_pe] cache、`dim=dc+dp`、`dim_v=dc`）→ 反吸收 GEMM。
- `examples/04_dsa/`：DSA（DeepSeek-V3.2 稀疏注意力）= lightning indexer
  （ReLU 头混合打分，GEMM/GEMV 双路径）→ 因果前缀精确 top-k → CLS 汇总
  token → gather + 无 pack M=1 稀疏行核（decode 专用小 M 内核的第一个实例）。
- `examples/05_csa/`：CSA（NSA 式三分支）= 压缩分支（块质心，可注入上游
  φ 缓存）+ 选块分支（**打分复用压缩分支 softmax 分数**、跨头求和、整块
  gather、固定激活首块/局部块）+ 窗口分支，sigmoid 门控求和；复用 04 的
  流式行核原语。
- `examples/06_kda/`：KDA（Kimi Delta Attention，门控 delta 规则线性注意
  力）双模式——naive 逐 token 递归（decode 路径，状态经 s0/s_out 跨调用
  传递）与 chunkwise 并行（块内 (I+N)⁻¹ 前代 + 注意力形状核；衰减核带
  精确因子化快路径 + 极端衰减回退）。
- 确定性契约贯穿所有组装：kv-split 关闭时单/多线程**逐位一致**；开启时按固定
  slice 顺序归并，跨运行**逐位一致**（DSA/CSA 的行路径无 split，天然逐位稳定）。

CMake 选项：`CPU_OPS_ENABLE_NATIVE`（默认 ON，`-march=native`）、
`CPU_OPS_BUILD_EXAMPLES`、`CPU_OPS_BUILD_TESTS`、`CPU_OPS_ISA_COMPILE_CHECK`
（默认 OFF；为 sapphirerapids / icelake-server 额外生成 OBJECT 编译目标，把
AVX512-BF16 / AMX 等非本机 ISA 的 kernel 纳入编译检查，不执行）。

作为依赖集成：`cmake --install build` 之后可 `find_package(cpu_ops)` 并链接
`cpu_ops::cpu_ops`（INTERFACE 目标，携带 include 路径与 Threads 依赖），或直接 vendor
`include/` 目录。头文件库意味着 kernel 在**你的**编译单元内实例化：请自带 ISA 编译选项
（如 `-O3 -march=native`）；无任何 SIMD 宏命中时自动走通用标量回退，功能正确但性能保底。

## 用法

### f32 / f64 GEMM

```cpp
#include <cpu_ops/cpu_ops.h>

using Gemm = cpu_ops::gemm::device::Gemm<
    float, cpu_ops::layout::RowMajor,    // A
    float, cpu_ops::layout::RowMajor,    // B
    float, cpu_ops::layout::RowMajor>;   // C/D

Gemm::Arguments args{
    {M, N, K},          // problem size
    {A, lda},           // A 矩阵及 leading dimension
    {B, ldb},
    {C, ldc},           // beta 项的源（可与 D 相同，原地累加）
    {D, ldd},           // 输出
    {alpha, beta}};     // epilogue 参数

Gemm gemm;
gemm(args);                 // 自动按硬件并发数多线程
gemm(args, 1);              // 强制单线程
```

`Arguments::split_k_slices`：0 = 自动（k ≥ 8·KC 的深归约自动切分，片数约 2×线程数，
受 k 深度和 4 MiB 工作区预算约束）；1 = 禁止；>1 = 强制片数。split-k 的两阶段归约
按固定顺序累加部分和，结果与单线程**逐位一致**。

### int8 量化 GEMM（VNNI / AMX）

```cpp
using GemmI8 = cpu_ops::gemm::device::GemmU8S8S32<
    cpu_ops::layout::RowMajor, cpu_ops::layout::RowMajor, cpu_ops::layout::RowMajor>;

GemmI8::Arguments args{{M, N, K}, {A_u8, lda}, {B_s8, ldb}, {C_s32, ldc}, {D_s32, ldd},
                       {alpha_s32, beta_s32}};
GemmI8 gemm;
gemm(args);
```

默认 policy 是 VNNI（x86）/ dotprod/i8mm（ARM）点积，任意平台有标量回退。
另有实验性 AMX-INT8 瓦片 policy（`-march=sapphirerapids` 编译、Linux 运行时
授权 tile；不满足时自动退回正确但慢的标量路径），opt-in 用法见
`gemm/threadblock/mma_policy_amx.h` 头注释。

### f16 / bf16 输入（f32 累加 / 输出）

```cpp
using GemmF16 = cpu_ops::gemm::device::GemmF16F32<
    cpu_ops::layout::RowMajor, cpu_ops::layout::RowMajor, cpu_ops::layout::RowMajor>;

// A/B 是 float16_t 数组，C/D/alpha/beta 仍是 float
GemmF16::Arguments args{{M, N, K}, {A_f16, lda}, {B_f16, ldb}, {C, ldc}, {D, ldd},
                        {alpha, beta}};
```

`float16_t` / `bfloat16_t` 见 `numeric_types.h`：从 float 构造做 RNE 舍入，隐式转回
float。`GemmBF16F32` 用法相同。八种 layout 组合均可直接实例化使用。

用 AVX512-BF16 编译（`-march=sapphirerapids` 等）时 `GemmBF16F32` 自动切换到原生
点积 policy（`vdpbf16ps`，打包流量减半、算力约为 f32 的 2×）；数值与 widen 路径不
逐位一致（bf16 乘积少一次舍入，反而更准）。显式传 `mma::WidenPolicy<bfloat16_t>` +
`GemmConfig<float>` 可强制旧的 widen 路径。

### MX 格式（mxfp8 / mxfp4 + E8M0 块缩放）

```cpp
using GemmMx = cpu_ops::gemm::device::GemmMxE4M3<cpu_ops::layout::RowMajor>;  // C layout

// A: m×k 行主序；B: k×n 列主序——两者的 k 都必须是连续维（"TN"）。
// 每 32 个连续 k 元素一个 E8M0 scale（scales[i * ld_scales + k/32]）。
cpu_ops::MxTensorRef<cpu_ops::fp8e4m3_t> a_ref{A_bytes, A_scales, lda, lda_scales};
cpu_ops::MxTensorRef<cpu_ops::fp8e4m3_t> b_ref{B_bytes, B_scales, ldb, ldb_scales};

GemmMx::Arguments args{{M, N, K}, a_ref, b_ref, {C, ldc}, {D, ldd}, {alpha, beta}};
GemmMx gemm;
gemm(args);
```

- 元素字节序：`fp8e4m3_t` / `fp8e5m2_t` 每元素 1 字节；`fp4e2m1_t` 每字节 2 个元素
  （k 偶数在低 nibble），此时 `ld` 必须为偶数。
- 编码器：`float16_t` 式 RNE 饱和编码（`fp8e4m3_t::from_float` 等），scale 用
  `e8m0_from_float`（舍入到最近的 2 的幂）。
- A/B 类型可以不同（如 e4m3×e5m2），C/D 固定 f32。
- 解码在打包循环内做（AVX2 下 e4m3 用 F16C 位技巧、e2m1 用 PSHUFB 查表；
  AVX512F 下同一套技巧 16 路宽，经 `_mm512_cvtph_ps` 转换），标量回退走
  constexpr 查找表。所有 SIMD 解码与标量**逐位一致**。
- 主循环为 f32 FMA：AVX-512F 主机上默认 512 位向量 + Fma512 tile
  （`detail::GemmMxDefaults`，Zen 4 实测比 AVX2 默认快 ~1.2×，与 f32 路径
  算力持平）；显式传 `Config` / `MmaPolicy` 模板参数可覆盖。
- opt-in VNNI 路径（`GemmMxE4M3Vnni` / `GemmMxE5M2Vnni` / `GemmMxE2M1Vnni`，或
  显式 `MmaPolicy = mma::MxVnniPolicy<SA, SB>`）：打包时把每个 32 元素块按块内
  最大值重量化为 int8（2 的幂量化子，A 侧 +128 偏移成 u8），主循环用
  `vpdpbusd` 字节点积——块内 int32 点积精确，逐块转 f32 按 wA·wB 缩放累加。
  e2m1 始终**精确**；e4m3/e5m2 块内动态范围超 int8 预算时小元素被舍入（每元素
  误差 ≤ amax/128，与同数据做逐块 int8 量化相当）；不传播 inf/nan。
  AVX512-VNNI 主机上默认 512 位 tile。
- 同格式与交叉格式组合都在调用处按需实例化（header-only），无预编译对象。

### 激活 / 归一化 / TopK（ops 层）

```cpp
#include <cpu_ops/cpu_ops.h>  // 或按需包含 cpu_ops/ops/*.h

cpu_ops::ops::gelu<float>(x, y, n);              // tanh 近似，f16/bf16 同签名
cpu_ops::ops::silu<float>(x, y, n, /*threads=*/4);
cpu_ops::ops::rmsnorm<float>(x, w, y, rows, cols, /*eps=*/1e-6f);
cpu_ops::ops::fused_add_rmsnorm<float>(x, res, w, y, res_out, rows, cols, eps);
// res_out = x + res；y = rmsnorm(res_out)·w——输入一遍读完（每线程 f32 行 scratch）
cpu_ops::ops::layernorm<float>(x, w, y, rows, cols, eps);
std::vector<int32_t> idx(k);
cpu_ops::ops::topk_indices(scores, n, k, idx.data());
// (值降序, 下标升序)，与 partial_sort 公式逐位一致；NaN 拒绝，k 截断到 n
```

- SIMD 数学底座（`simd::tanh/sigmoid/silu/gelu`）全部组合在 `exp2` 多项式上，
  任意 Vec 宽度自动获得 ISA 快路径；窄存的输出走向量 RNE 转换
  （f16 `FCVT`，bf16 整数技巧 `(u+0x7FFF+lsb)>>16`）。
- 归一化行统计用双累加器 + 固定归约树；任务为 p×8 个连续大块
  （`parallel_for` 一波的同步开销几十 µs，细粒度分波会被吃掉）。
- topk 为 MSD 字节级 radix-select（-0 规范化到 +0 保平局规则），DSA 的
  选集阶段直接调用。

### 融合 epilogue

```cpp
#include <cpu_ops/gemm/device/gemm_fused.h>

using namespace cpu_ops;
using Fused = gemm::device::GemmFusedF32<
    layout::RowMajor, layout::RowMajor, layout::RowMajor,
    epilogue::BiasAdd<float>, epilogue::Relu>;

Fused::Arguments args{{M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {D, ldd},
                      {alpha, beta,
                       epilogue::Chain<epilogue::BiasAdd<float>, epilogue::Relu>{
                           epilogue::BiasAdd<float>{bias_ptr}, epilogue::Relu{}}}};
Fused gemm;
gemm(args);   // D = relu(alpha·A·B + beta·C + bias[col])，op 在输出瓦片还在寄存器时融合
```

内置 op：`Relu`、`Gelu`（tanh 近似）、`Silu`、`Clamp<T>`、`BiasAdd<T>`（按列）、
`ScalePerRow<T>`、`ScalePerCol<T>`；`Chain<Ops...>` 顺序组合。op 契约是位置感知的
`(x, row, col)` + `apply_vec` 车道形式，可自由扩展。包含
`gemm_fused.h` 即在自己的编译单元内实例化整个融合 kernel。

### Attention / MLA / DSA / CSA / KDA（组装示例）

fused attention、MLA、DSA、CSA 与 KDA 的完整用法、组装方式与扩展缝见
`examples/02_fused_attention/README.md` … `examples/06_kda/README.md`。要点：

```cpp
#include "examples/02_fused_attention/attention.h"   // 或按例子目录引用
cpu_ops::attention::Attention<float>::Arguments args;  // 步长化 GQA 入口
args.heads_q = 32; args.heads_kv = 8; /* ... */        // MHA: 相等；MQA: 1
args.split_kv_slices = 0;                              // decode 自动 KV-split
```

MLA 在其上多两层 GEMM 编排（吸收/反吸收），核心调用是
`heads_kv=1、dim=dc+dp（如 576）、dim_v=dc（如 512）`，K/V 指向同一份
`[seq_kv × (dc+dp)]` 共享 latent cache。DSA 复用同一潜空间：indexer 打分
`Σ_j w_j·ReLU(q_idx_j·k_idx)` → 因果前缀精确 top-k → CLS 汇总行 →
对 gather 出的 (k+1) 行做吸收式稀疏注意力（`DsaAttention<T>`）。

### 通用说明

- f32/f64 与 int8 各自的 A/B/C 八种 RowMajor/ColumnMajor 组合均可直接实例化。
- C 与 D 可指向同一缓冲区（原地 `C = alpha·A·B + beta·C`）。
- 线程数只改变任务划分，不改变每个输出元素的运算顺序：单线程与多线程结果
  **逐位一致**（int8 是精确整数；f32/f64/f16/bf16/MX 的浮点求和顺序固定——split-k
  的部分和也按 slice 顺序归约）。
- 自定义 tile 配置 / epilogue / mma policy：`gemm/device/gemm.h` 自带完整定义，
  直接在自己的编译单元实例化 `Gemm<...>` 特化即可。

## 数据类型

| 入口 | A×B → C/D（累加） | 计算路径 |
|---|---|---|
| `Gemm<float, ...>` | f32×f32 → f32 | f32 FMA |
| `Gemm<double, ...>` | f64×f64 → f64 | f64 FMA |
| `GemmU8S8S32` | u8×s8 → s32 | VNNI `vpdpbusd`（4 路字节点积）；ARM dotprod/i8mm；AMX-INT8 policy 可选（实验性） |
| `GemmF16F32` / `GemmBF16F32` | f16/bf16 → f32（f32 累加） | 打包时转 f32 走 FMA；bf16 在 AVX512-BF16 上默认走 `vdpbf16ps` 原生点积 |
| `GemmMxE4M3` / `GemmMxE5M2` / `GemmMxE2M1` | mxfp8/mxfp4 → f32 | 打包时 SIMD 解码 ×E8M0 块缩放，走 f32 FMA（AVX-512F 上默认 512 位主循环） |
| `GemmMxE4M3Vnni` / `GemmMxE5M2Vnni` / `GemmMxE2M1Vnni` | mxfp8/mxfp4 → f32 | opt-in：块内 int8 重量化 + `vpdpbusd` 点积 + 逐块 f32 缩放（e2m1 精确） |
| `attention::Attention<T>`（示例） | f32/f16/bf16 → 同型 | QKᵀ + online softmax + PV 融合，f32 累加 |
| `attention::MlaAttention<T>`（示例） | 共享 latent cache → 每 head 输出 | 吸收式：2×GEMM + fused 核心（dim_v ≠ dim） |
| `attention::DsaAttention<T>`（示例） | latent cache + indexer → 每 head 输出 | 稀疏式：indexer 打分 + 精确 top-k + CLS + 无 pack 行核 |
| `attention::CsaAttention<T>`（示例） | latent cache（+压缩缓存）→ 每 head 输出 | 三分支：压缩/选块（top-n 块）/窗口，sigmoid 门控求和 |
| `attention::KdaAttention<T>`（示例） | q/k/v + β/λ logits（+初始状态）→ 输出 + 终态 | 线性注意力：门控 delta 规则，naive/chunk 双模式 |
| `ops::gelu/silu<T>` | T → T | f32 车道数学 + 向量 RNE 窄存 |
| `ops::rmsnorm / fused_add_rmsnorm / layernorm<T>` | [rows×cols] → 同型 | 双累加器行统计，fused 输入一遍读完 |
| `ops::topk_indices` | f32 → int32[k] | MSD radix-select，与 partial_sort 逐位同语义 |

f16 / bf16 / MX 路径的收益来自**操作数内存流量减半 / 减四**以及更小的 cache 占用；
乘加本身仍以 f32 FMA 执行，算力上限与 f32 路径相同（两个例外：AVX512-BF16 目标上
bf16 默认改用 `vdpbf16ps`，每条指令完成 32 次乘加，峰值约为 f32 的 2×——本机无
该指令集，未实测；MX 可 opt-in `MxVnniPolicy` 改用 `vpdpbusd` int8 点积，峰值
同样约为 f32 的 2×，Zen 4 实测见下）。

## 性能

参考数据：Intel Core Ultra 7 258V（4 P 核 + 4 E 核，AVX2+FMA+AVX-VNNI），g++ 15.2，
`-O3 -march=native`，行主序，best-of-5；对比基准 OpenBLAS 0.3.32（Haswell kernel）。
多线程数据在绑定核心的条件下测得，避免调度噪声。

f32 方阵 GFLOPS（cpu_ops）及与 OpenBLAS 之比：

| size | 1T | 4T | 8T | vs OpenBLAS 1T | vs OpenBLAS 8T |
|---|---|---|---|---|---|
| 512³  | 128 | 257 | 503 | ~1.0x | 1.02x |
| 1024³ | 125 | 444 | 495 | 0.92x | 1.00x |
| 2048³ | 123 | 425 | 518 | 0.91x | 0.93x |
| 4096³ | 116 | 402 | 511 | 0.82x | 0.93x |

8 线程时双方都触及该机 f32 算力上限（~500–550 GFLOPS），大体持平。

不规则形状（8T，vs OpenBLAS）：

| 形状 | cpu_ops | OpenBLAS | 比值 |
|---|---|---|---|
| 4096×64×1024（tall-skinny） | 416 | 182 | 2.28x |
| 32×32×8192（深归约，split-k 生效） | 325 | 197 | 1.65x |
| 1000×999×997（全奇数） | 476 | 522 | 0.91x |
| 8192×128×128（LLM 投影形） | 478 | 494 | 0.97x |
| 1×4096×4096（GEMV） | 21 | 21 | 1.05x |

split-k 效果（8T，`split_k_slices` 强制值对比，GFLOPS）：

| 形状 | s=1（关） | auto | 最佳强制值 |
|---|---|---|---|
| 16×16×16384 | 67 | 173 | 199（s=8） |
| 32×32×8192 | 147 | 280 | 243（s=4） |
| 1024³ | 490 | 492（auto 不切） | — |

数据类型（GFLOPS，同机 f32 为基准）：

| 形状 | f32 | f16 | bf16 | e4m3 | e2m1 |
|---|---|---|---|---|---|
| 1024³ 8T | 460–473 | 480 | 494 | 486 | 495 |
| 8192×128×128 8T | 420–424 | 456 | 398 | 458 | 441 |
| 8192×4096×128 8T | 349 | 372 | — | 363 | 373 |

低精度类型在小 K / 大 M 形状上领先 f32 最多 ~15%（省内存流量），方阵打平
（都已算力受限）；深 k 极小输出（32×32×8192）MX 路径有解码开销，比 f32 慢
~20–30%。

MX-VNNI opt-in 路径（Ryzen 9 7900X，Zen 4 AVX512-VNNI，GFLOPS）：

| size | e4m3 1T / 8T | e5m2 1T / 8T | e2m1 1T / 8T |
|---|---|---|---|
| 1024³ | 269 / 798 | 273 / 868 | 272 / 817 |
| 2048³ | 286 / 1490 | 287 / 1372 | 292 / 1535 |

同机 MX f32 路径 2048³ 为 133 / 971（e4m3），即 VNNI 路径 1T ~2.1×、8T
~1.5×；8T 下同时超过 bf16 原生 `vdpbf16ps`（1138）与 f32 FMA（964）。

融合 epilogue（2048³ 8T，bias+ReLU）：与无融合的 GEMM 相比开销 ≈ 0；比
「GEMM + 单独逐元素 pass」快 ~8%（省一遍 C 的读写扫描）。

int8（u8×s8→s32）GOPS（一个乘积累加计 2 ops）：

| size | 1T | 2T | 4T | 8T |
|---|---|---|---|---|
| 1024³ | 175.6 | 413.2 | 412.3 | 715.7 |
| 2048³ | 179.7 | 408.4 | 821.3 | 772.6 |

参照：朴素三重循环（ikj 序，编译器自动向量化）单线程 512³ 约 30–46 GFLOPS。

ops 层（M1 Pro，4T，8192×7168 / 64M 元素）：rmsnorm 4.86 ms vs 朴素 62.7 ms
（**12.9×**，~145 GB/s）；fused_add_rmsnorm 12.4 ms（~76 GB/s）；silu f32
53 GB/s（vs 朴素 11.9×）、bf16 20 GB/s（向量 RNE 窄存后 2.5×）；topk
n=64k/k=2048 467 µs（vs partial_sort 1.3×，更大 n 优势扩大）。

Fused attention（Apple M1 Pro，4T，H=16 d=128 causal f32）：seq 512/1024/2048
分别 109 / 144 / 181 GFLOPS——约为同机纯 f32 GEMM 吞吐的 74%；bf16 操作数再快
~3%。MLA prefill（H=32，dn/dp/dc=128/64/512，f32）seq 1024/2048 为 230 /
740 ms。稠密 decode（seq_q=1）目前走通用面板路径：GQA ~4.5 GB/s，MLA ~0.3
GB/s（rows=1 时 K 面板打包成本无法摊销，且各 head 重复打包同一份共享 K）。

DSA decode（同机 4T，B=1 H=32，dn/dp/dc=128/64/512，indexer 64×128，
topk=2048，bf16，对同形稠密 MLA）：

| kv len | MLA 4T ms | DSA 4T ms | 加速比 | 有效 GB/s |
|---|---|---|---|---|
| 8192 | 30.2 | 5.2 | 5.8x | 29.7 |
| 32768 | 117.5 | 10.6 | 11.1x | 55.5 |
| 65536 | 228.8 | 17.3 | 13.3x | 67.2 |

CSA decode（同机同形，NSA 默认 block=64、top-16 块、window=512）：

| kv len | MLA 4T ms | CSA 4T ms | 加速比 |
|---|---|---|---|
| 8192 | 30.8 | 3.7 | 8.3x |
| 32768 | 115.6 | 7.8 | 14.8x |
| 65536 | 230.3 | 13.3 | 17.3x |

CSA 的块打分直接复用压缩分支（64k 下每头仅 1024 行质心点积），decode
耗时随上下文增长极缓；DSA 的独立 indexer 每 token 要扫 64×128 的 key
流——两者是 token 粒度精确选集 vs 块粒度近似选集的结构取舍。稀疏路径
把每 token 的 KV 触达从整条 cache 降到几百~几千行且全部无 pack顺流。DSA
prefill（512×8192）为 0.34×、CSA 为 0.53×：逐行选集语义使注意力只能以
M=1 行核算力执行（见「已知限制」）。（数字来自 `examples/*_attention|--bench`
、`examples/04_dsa/dsa --bench` 与 `examples/05_csa/csa --bench`）

### 已知限制

- GEMV 无专用 kernel，单线程 GEMV 约为 OpenBLAS 的 0.5x。
- 单线程 kernel 相对 OpenBLAS 有 0.82–0.92x 的差距，来自微内核调优深度。
- ARM 路径（NEON / SVE2 / SME）尚未在大规模部署中验证，SME 为实验性。
- AMX-INT8 路径无对应硬件，仅经编译检查与算法级等效验证（标量复刻），未在
  真机运行。AVX512-BF16 / AVX512F / AVX512-VNNI（f32/f16 512 位主循环、int8
  512 位 VNNI、MX 16 路解码 + 512 位主循环、MX-VNNI int8 点积）已在
  Ryzen 9 7900X（Zen 4）实测。
- 稠密 attention decode（seq_q 很小）复用通用查询面板路径，带宽利用率低
  （MLA 下尤甚）。稀疏行核的窄存→f32 已有向量宽化原语（`simd::widen_f16/
  widen_bf16`，NEON `vcvt`/`vshll`、AVX2 F16C/移位）+ indexer 按 8 头分块
  （k_idx 流量 ÷8）+ dot_f32 四累加器展开（打破 fma 单链延迟——工作集扫描
  证明行核是指令吞吐受限而非缓存受限，展开后单核流式点积 12.6→40-52
  GB/s）——DSA decode 64k 达 13.3x / 67 GB/s，CSA 17.3x；DSA prefill 提至
  0.60x、CSA prefill 0.95x（逼近稠密持平）。剩余限制：每 token 串行的
  top-k/softmax pass 与 gather 流量。
- DSA/CSA prefill 保持逐行精确选集语义，注意力阶段只能 M=1 行核执行
  （DSA 0.60x / CSA 0.95x @512×8192，dot 展开后已逼近稠密）；查询面板共享
  选集重回 GEMM 核是后续工作。DSA 的 FP8 indexer 在 CPU 上的对应物（int8
  点积原语）待接。
- KDA chunked 的块内六矩阵运算已走库 GEMM 原语（衰减行向量 exp2 构建），
  对 naive 提速 ~1.45×（8192: 446→307 ms，4T）；前代求解与极端衰减
  （|G|>140/块）保留标量精确路径。
- MLA→DSA→CSA→KDA 路线已全部完成（`examples/02`–`06`）；后续为性能项（见
  上）与 GEMV / 窄存向量加载等既有缺口。

## 架构

| 层 | 组件 | 职责 |
|---|---|---|
| 指令层 | `simd::Vec<T, N>` | SIMD 寄存器抽象：`load / store / set1 / fmadd / dpbusd / dpbf16ps` |
| 微内核层 | `MmaAtom` / `MmaAtomVnni` / `MmaAtomBf16` / `MmaAtomSmeF32` / `MmaAtomAmxInt8` | MR×NR 累加瓦片驻留寄存器（或 SME ZA / AMX 瓦片） |
| 策略层 | `mma::FmaPolicy` / `VnniPolicy` / `WidenPolicy` / `MxPolicy` / `Bf16Policy` / `SmePolicyF32` / `AmxPolicy` | 定义 ElemA/ElemB/PackedA/PackedB/AccT、k 步长、打包布局、微内核类型——`BlockGemm` 由 policy 驱动，扩展新指令集或新数据类型只需新增 policy |
| 打包层 | `gemm::threadblock::pack_a / pack_b` | A/B 面板按微瓦片交织成连续内存，零填充处理边缘；Packed 类型 ≠ 输入类型时在打包处转换 |
| 分块层 | `gemm::threadblock::BlockGemm<Policy, ...>` | MC/NC/KC 三级 cache 分块主循环；B 条带驻留 L1，A 面板流式复用 |
| 线程层 | `thread::ThreadPool` + `gemm::threadblock::partition_gemm` | 持久线程池；输出按 MR/NR 对齐的矩形区域静态划分（先切 N，不足再切 M），k 方向可再切 split-k 片 |
| 入口层 | `gemm::device::Gemm` / `GemmMx` | 类模板入口：`Arguments` + `operator()` |
| Epilogue | `epilogue::LinearCombination` / `LinearCombinationFused` | 写回时融合 `D = alpha·acc + beta·C` 及 op 链；split-k 时用 `PartialSum` 暂存部分和 |

微内核默认 `MR=6, NR=16`（f32/AVX2：12 个 YMM 累加器 + 2 个 B 向量 + 1 个广播，
打满 16 个向量寄存器）；cache 块默认 `MC=126, NC=256, KC=256`。

## ISA 支持

同一套源码编译期分派（`include/cpu_ops/arch/simd.h` 检测，无运行时分派）：

| 平台 | 数据类型 | 指令 |
|---|---|---|
| x86-64 AVX2+FMA | f32 / f64 / f16 / bf16 / mxfp8 / mxfp4 | `vfmadd`（f16 转换用 F16C；MX 解码用 F16C+PSHUFB 快路径） |
| x86 AVX-VNNI / AVX512-VNNI | u8×s8→s32；mxfp8/mxfp4（opt-in `MxVnniPolicy`） | `vpdpbusd`（AVX512-VNNI 下默认 512 位 tile；MX 为块内 int8 重量化 + 逐块 f32 缩放） |
| x86 AVX512F | f32 / f16→f32 / mxfp8 / mxfp4 | f32 主循环默认 512 位 FMA（f32/f16/MX，Zen 4 实测）；MX 解码 `_mm512_cvtph_ps` 16 路宽 |
| x86 AVX512-BF16 | bf16→f32 | `vdpbf16ps`（`GemmBF16F32` 默认；Zen 4 实测） |
| x86 AMX-INT8（实验性，opt-in） | u8×s8→s32 | `tdpbusd` 16×16 瓦片（编译验证 + 打包布局经标量回退精确验证；真机未运行） |
| ARM NEON | f32 | `vfmaq_f32` |
| ARM dotprod / i8mm | u8×s8→s32 | `vdotq_u32`（偏置技巧）/ `vusdotq_s32`（qemu 探针逐位验证） |
| ARM SVE/SVE2 定长（`-msve-vector-bits=N`） | f32 / f64 | `svmla` |
| ARM SME（实验性） | f32 | `svfmopa`（外积累加进 ZA 瓦片） |
| 任意平台 | 全部 | 通用标量回退（`CPU_OPS_FORCE_SCALAR` 可强制） |

int8 默认路径选择 `vpdpbusd`（VNNI）而非 AMX：AMX 只支持 int8/bf16/fp16/fp8、不支持
f32/f64，且仅存在于部分至强；VNNI 在消费级 CPU 上普遍具备。AMX-INT8 以实验性
opt-in policy 提供（`mma_policy_amx.h`）：需要 `-mamx-tile -mamx-int8` 编译与 Linux
`arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` 运行时授权（首次使用自动
申请）；不满足时 atom 退回标量路径，结果保持精确。

ARM 构建（需要 aarch64 工具链，GCC 14+ / Clang 18+）：

```bash
# NEON（默认 AArch64 即可）
aarch64-linux-gnu-g++ -O3 ...
# SVE2 定长（如 256-bit）
aarch64-linux-gnu-g++ -O3 -march=armv9-a+sve2 -msve-vector-bits=256 ...
# SME（实验性；kernel 在你的编译单元内实例化）
aarch64-linux-gnu-g++ -O3 -march=armv9.2-a+sme -msve-vector-bits=512 your_kernel.cc ...
# int8 dotprod / i8mm（默认 VnniPolicy 自动启用）
aarch64-linux-gnu-g++ -O3 -march=armv8.2-a+dotprod ...   # 或 armv8.6-a（含 i8mm）
```

x86 ISA 编译检查（本机无 AVX-512/AMX 时保持这些头文件编译干净）：

```bash
cmake -B build -DCPU_OPS_ISA_COMPILE_CHECK=ON && cmake --build build -j
```

## 目录结构

```
include/cpu_ops/
├── cpu_ops.h                        # 伞头文件
├── status.h                         # Status 枚举
├── numeric_types.h                  # float16_t / bfloat16_t
├── mx_formats.h                     # fp8e4m3_t / fp8e5m2_t / fp4e2m1_t / E8M0 / MxTensorRef
├── matrix_shape.h                   # Shape（静态 M/N/K）
├── gemm_coord.h                     # GemmCoord（运行时 problem size）
├── tensor_ref.h                     # TensorRef（非拥有 2D 视图 + leading dimension）
├── layout/
│   └── matrix.h                     # RowMajor / ColumnMajor
├── arch/                            # ISA 级原语
│   ├── simd.h                       # ISA 检测 + simd::Vec（AVX2/VNNI/AVX512/SVE/NEON/标量）
│   └── mma_atom.h                   # FMA 寄存器微内核
├── gemm/
│   ├── device/                      # device 级算子（完整定义，包含即用）
│   │   ├── gemm.h                   # device::Gemm、GemmConfig、U8S8S32/F16/BF16 别名
│   │   ├── gemm_fused.h             # GemmFusedF32/F64 融合 epilogue 别名
│   │   └── gemm_mx.h                # device::GemmMx、E4M3/E5M2/E2M1 别名
│   ├── kernel/
│   │   └── run_blocked.h            # 线程划分 + split-k 两阶段编排（Gemm/GemmMx 共享）
│   └── threadblock/                 # 线程块级：policy、打包、主循环、区域划分
│       ├── block_gemm.h             # policy 驱动的 cache 分块主循环 + 宏内核
│       ├── mma_policy_fma.h         # f32/f64 policy
│       ├── mma_policy_vnni.h        # int8 点积 policy + 微内核 + 打包
│       ├── mma_policy_widen.h       # f16/bf16 → f32 打包转换 policy
│       ├── mma_policy_mx.h          # MX 格式解码 policy（AVX2 / AVX512F SIMD 解码快路径）
│       ├── mma_policy_mx_vnni.h     # MX opt-in VNNI policy：块内 int8 重量化 + vpdpbusd
│       ├── mma_policy_bf16.h        # AVX512-BF16 原生点积 policy（vdpbf16ps）+ 打包
│       ├── mma_policy_amx.h         # AMX-INT8 瓦片 policy（实验性，opt-in，含标量回退）
│       ├── mma_policy_sme.h         # ARM SME FMOPA policy（实验性）
│       ├── pack.h                   # A/B 面板打包（零填充，DstT 独立于输入类型）
│       └── threadblock_swizzle.h    # 输出区域划分（含 k 方向 split-k 切片）
├── epilogue/
│   ├── linear_combination.h         # LinearCombination (alpha/beta/ReLU)
│   ├── fusion.h                     # 融合 op 链：Relu/Clamp/BiasAdd/ScalePerRow/ScalePerCol/Chain
│   └── partial_sum.h                # split-k 第一阶段的部分和 epilogue
├── ops/                             # 算子原语层
│   ├── activation.h                 # gelu/silu（f32 车道 + 向量窄存）
│   ├── norm.h                       # rmsnorm / fused add+rmsnorm / layernorm
│   ├── topk.h                       # MSD radix-select 精确 top-k
│   └── detail/parallel.h            # p×8 连续大块的任务切分
└── thread/
    └── thread_pool.h                # 持久线程池（inline 实现）
examples/
├── 00_basic_gemm.cc                 # 最小 GEMM 示例
├── 01_benchmark.cc                  # GEMM 基准
├── 02_fused_attention/              # 组装示例：fused attention（含自校验+基准 main）
│   ├── attention_kernel.h           # fused 块内核：Q 驻留 + online softmax + P·V
│   ├── attention.h                  # Attention<T>（GQA/MHA/MQA）+ CSA/KDA/DSA 扩展缝
│   ├── fused_attention.cc           # 自校验 + --bench
│   ├── CMakeLists.txt  README.md
└── 03_mla/                          # 组装示例：吸收式 MLA（同构）
    ├── mla.h  mla.cc  CMakeLists.txt  README.md
└── 04_dsa/                          # 组装示例：DSA 稀疏注意力（同构）
    ├── dsa.h  dsa.cc  CMakeLists.txt  README.md
└── 05_csa/                          # 组装示例：CSA（NSA 式三分支，复用 04 行核）
    ├── csa.h  csa.cc  CMakeLists.txt  README.md
└── 06_kda/                          # 组装示例：KDA（naive/chunk 双模式）
    ├── kda.h  kda.cc  CMakeLists.txt  README.md
test/      核心单元测试（attention/mla/dsa/csa/kda 的校验在各自示例 main 中）
```
