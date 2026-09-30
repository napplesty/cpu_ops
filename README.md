# cpu-ops

CPU 高性能 GEMM 模板库：SIMD 微内核 + cache 分块 + 多线程。C++17，无第三方依赖，
按 `.h` 声明 + `.cc` 显式实例化组织，编译为静态库 `libcpu_ops.a`。

- f32 / f64 GEMM：FMA 主路径
- int8 量化 GEMM（u8×s8→s32）：AVX-VNNI 点积指令
- f16 / bf16 输入 GEMM（f32 累加 / 输出）：打包时转 f32
- MX 格式 GEMM（mxfp8 e4m3/e5m2、mxfp4 e2m1 + E8M0 块缩放）：解码后走 f32 FMA
- Split-K：深 k 自动切分 + 确定性两阶段归约
- 融合 epilogue：alpha/beta 之后按位置融合的 op 链（ReLU / Clamp / BiasAdd / …）
- ARM 支持：NEON / SVE2 定长 / SME（实验性）
- 任意平台均有标量回退；单线程与多线程结果**逐位一致**

## 构建

```bash
cmake -B build && cmake --build build -j
ctest --test-dir build          # 11 个测试：simd / gemm / int8 / f16 / mx / fusion + 标量回退
./build/basic_gemm              # 最小示例
./build/benchmark               # GFLOPS / GOPS 基准
```

CMake 选项：`CPU_OPS_ENABLE_NATIVE`（默认 ON，`-march=native`）、
`CPU_OPS_BUILD_EXAMPLES`、`CPU_OPS_BUILD_TESTS`。

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

### int8 量化 GEMM（VNNI）

```cpp
using GemmI8 = cpu_ops::gemm::device::GemmU8S8S32<
    cpu_ops::layout::RowMajor, cpu_ops::layout::RowMajor, cpu_ops::layout::RowMajor>;

GemmI8::Arguments args{{M, N, K}, {A_u8, lda}, {B_s8, ldb}, {C_s32, ldc}, {D_s32, ldd},
                       {alpha_s32, beta_s32}};
GemmI8 gemm;
gemm(args);
```

### f16 / bf16 输入（f32 累加 / 输出）

```cpp
using GemmF16 = cpu_ops::gemm::device::GemmF16F32<
    cpu_ops::layout::RowMajor, cpu_ops::layout::RowMajor, cpu_ops::layout::RowMajor>;

// A/B 是 float16_t 数组，C/D/alpha/beta 仍是 float
GemmF16::Arguments args{{M, N, K}, {A_f16, lda}, {B_f16, ldb}, {C, ldc}, {D, ldd},
                        {alpha, beta}};
```

`float16_t` / `bfloat16_t` 见 `element_types.h`：从 float 构造做 RNE 舍入，隐式转回
float。`GemmBF16F32` 用法相同。八种 layout 组合均已实例化。

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
- 解码在打包循环内做（AVX2 下 e4m3 用 F16C 位技巧、e2m1 用 PSHUFB 查表），
  标量回退走 constexpr 查找表。
- 同格式组合 × Row/Col C 已实例化于 `src/gemm_mx.cc`；交叉格式请包含
  `cpu_ops/detail/gemm_device_defn.h` 自行实例化。

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

内置 op：`Relu`、`Clamp<T>`、`BiasAdd<T>`（按列）、`ScalePerRow<T>`、
`ScalePerCol<T>`；`Chain<Ops...>` 顺序组合。op 契约是位置感知的
`(x, row, col)` + `apply_vec` 车道形式，可自由扩展。融合 epilogue 不在静态库中
预实例化——包含 `gemm_fused.h` 即在自己的编译单元内联整个 kernel。

### 通用说明

- f32/f64 与 int8 各自的 A/B/C 八种 RowMajor/ColumnMajor 组合均已实例化。
- C 与 D 可指向同一缓冲区（原地 `C = alpha·A·B + beta·C`）。
- 线程数只改变任务划分，不改变每个输出元素的运算顺序：单线程与多线程结果
  **逐位一致**（int8 是精确整数；f32/f64/f16/bf16/MX 的浮点求和顺序固定——split-k
  的部分和也按 slice 顺序归约）。
- 自定义 tile 配置 / epilogue / mma policy：包含 `cpu_ops/detail/gemm_device_defn.h`
  并显式实例化自己的 `Gemm<...>` 特化。

## 数据类型

| 入口 | A×B → C/D（累加） | 计算路径 |
|---|---|---|
| `Gemm<float, ...>` | f32×f32 → f32 | f32 FMA |
| `Gemm<double, ...>` | f64×f64 → f64 | f64 FMA |
| `GemmU8S8S32` | u8×s8 → s32 | VNNI `vpdpbusd`（4 路字节点积） |
| `GemmF16F32` / `GemmBF16F32` | f16/bf16 → f32（f32 累加） | 打包时转 f32，走 f32 FMA |
| `GemmMxE4M3` / `GemmMxE5M2` / `GemmMxE2M1` | mxfp8/mxfp4 → f32 | 打包时 SIMD 解码 ×E8M0 块缩放，走 f32 FMA |

f16 / bf16 / MX 路径的收益来自**操作数内存流量减半 / 减四**以及更小的 cache 占用；
乘加本身仍以 f32 FMA 执行，算力上限与 f32 路径相同。

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

融合 epilogue（2048³ 8T，bias+ReLU）：与无融合的 GEMM 相比开销 ≈ 0；比
「GEMM + 单独逐元素 pass」快 ~8%（省一遍 C 的读写扫描）。

int8（u8×s8→s32）GOPS（一个乘积累加计 2 ops）：

| size | 1T | 2T | 4T | 8T |
|---|---|---|---|---|
| 1024³ | 175.6 | 413.2 | 412.3 | 715.7 |
| 2048³ | 179.7 | 408.4 | 821.3 | 772.6 |

参照：朴素三重循环（ikj 序，编译器自动向量化）单线程 512³ 约 30–46 GFLOPS。

### 已知限制

- GEMV 无专用 kernel，单线程 GEMV 约为 OpenBLAS 的 0.5x。
- 单线程 kernel 相对 OpenBLAS 有 0.82–0.92x 的差距，来自微内核调优深度。
- ARM 路径（NEON / SVE2 / SME）尚未在大规模部署中验证，SME 为实验性。

## 架构

| 层 | 组件 | 职责 |
|---|---|---|
| 指令层 | `simd::Vec<T, N>` | SIMD 寄存器抽象：`load / store / set1 / fmadd / dpbusd` |
| 微内核层 | `MmaAtom` / `MmaAtomVnni` / `MmaAtomSmeF32` | MR×NR 累加瓦片驻留寄存器（或 SME ZA 瓦片） |
| 策略层 | `mma::FmaPolicy` / `VnniPolicy` / `WidenPolicy` / `MxPolicy` / `SmePolicyF32` | 定义 ElemA/ElemB/PackedA/PackedB/AccT、k 步长、打包布局、微内核类型——`BlockGemm` 由 policy 驱动，扩展新指令集或新数据类型只需新增 policy |
| 打包层 | `detail::pack_a / pack_b` | A/B 面板按微瓦片交织成连续内存，零填充处理边缘；Packed 类型 ≠ 输入类型时在打包处转换 |
| 分块层 | `BlockGemm<Policy, ...>` | MC/NC/KC 三级 cache 分块主循环；B 条带驻留 L1，A 面板流式复用 |
| 线程层 | `ThreadPool` + `partition_gemm` | 持久线程池；输出按 MR/NR 对齐的矩形区域静态划分（先切 N，不足再切 M），k 方向可再切 split-k 片 |
| 入口层 | `gemm::device::Gemm` / `GemmMx` | 类模板入口：`Arguments` + `operator()` |
| Epilogue | `epilogue::LinearCombination` / `LinearCombinationFused` | 写回时融合 `D = alpha·acc + beta·C` 及 op 链；split-k 时用 `PartialSum` 暂存部分和 |

微内核默认 `MR=6, NR=16`（f32/AVX2：12 个 YMM 累加器 + 2 个 B 向量 + 1 个广播，
打满 16 个向量寄存器）；cache 块默认 `MC=126, NC=256, KC=256`。

## ISA 支持

同一套源码编译期分派（`include/cpu_ops/detail/simd.h` 检测，无运行时分派）：

| 平台 | 数据类型 | 指令 |
|---|---|---|
| x86-64 AVX2+FMA | f32 / f64 / f16 / bf16 / mxfp8 / mxfp4 | `vfmadd`（f16 转换用 F16C；MX 解码用 F16C+PSHUFB 快路径） |
| x86 AVX-VNNI / AVX512-VNNI | u8×s8→s32 | `vpdpbusd` |
| ARM NEON | f32 | `vfmaq_f32` |
| ARM SVE/SVE2 定长（`-msve-vector-bits=N`） | f32 / f64 | `svmla` |
| ARM SME（实验性） | f32 | `svfmopa`（外积累加进 ZA 瓦片） |
| 任意平台 | 全部 | 通用标量回退（`CPU_OPS_FORCE_SCALAR` 可强制） |

int8 路径选择 `vpdpbusd`（VNNI）而非 AMX：AMX 只支持 int8/bf16/fp16/fp8、不支持
f32/f64，且仅存在于部分至强；VNNI 在消费级 CPU 上普遍具备。

ARM 构建（需要 aarch64 工具链，GCC 14+ / Clang 18+）：

```bash
# NEON（默认 AArch64 即可）
aarch64-linux-gnu-g++ -O3 ...
# SVE2 定长（如 256-bit）
aarch64-linux-gnu-g++ -O3 -march=armv9-a+sve2 -msve-vector-bits=256 ...
# SME（实验性）
aarch64-linux-gnu-g++ -O3 -march=armv9.2-a+sme -msve-vector-bits=512 src/gemm_sme_f32.cc ...
```

## 目录结构

```
include/cpu_ops/
├── cpu_ops.h                        # 伞头文件
├── matrix_shape.h                   # Shape / GemmCoord
├── layout.h                         # RowMajor / ColumnMajor / TensorRef
├── element_types.h                  # float16_t / bfloat16_t
├── mx_formats.h                     # fp8e4m3_t / fp8e5m2_t / fp4e2m1_t / E8M0 / MxTensorRef
├── status.h                         # Status 枚举
├── epilogue/
│   ├── linear_combination.h         # LinearCombination (alpha/beta/ReLU)
│   ├── fusion.h                     # 融合 op 链：Relu/Clamp/BiasAdd/ScalePerRow/ScalePerCol/Chain
│   └── partial_sum.h                # split-k 第一阶段的部分和 epilogue
├── gemm/device/
│   ├── gemm.h                       # device::Gemm 声明、GemmConfig、U8S8S32/F16/BF16 别名
│   ├── gemm_fused.h                 # GemmFusedF32/F64（头文件实例化路径）
│   └── gemm_mx.h                    # device::GemmMx 声明、E4M3/E5M2/E2M1 别名
└── detail/                          # 内部实现（模板机制）
    ├── simd.h                       # ISA 检测 + simd::Vec（AVX2/VNNI/SVE/NEON/标量）
    ├── mma_atom.h                   # FMA 寄存器微内核
    ├── mma_policy_fma.h             # f32/f64 policy
    ├── mma_policy_vnni.h            # int8 点积 policy + 微内核 + 打包
    ├── mma_policy_widen.h           # f16/bf16 → f32 打包转换 policy
    ├── mma_policy_mx.h              # MX 格式解码 policy（含 AVX2 SIMD 解码快路径）
    ├── mma_policy_sme.h             # ARM SME FMOPA policy（实验性）
    ├── pack.h                       # A/B 面板打包（零填充，DstT 独立于输入类型）
    ├── block_gemm.h                 # policy 驱动的 cache 分块主循环 + 宏内核
    ├── thread_pool.h                # 持久线程池
    ├── thread_swizzle.h             # 输出区域划分（含 k 方向 split-k 切片）
    └── gemm_device_defn.h           # Gemm/GemmMx 共享编排：分区、split-k、归约
src/
├── thread_pool.cc
├── thread_swizzle.cc
├── gemm_f32.cc                      # float × 8 种 layout 组合显式实例化
├── gemm_f64.cc                      # double × 8
├── gemm_vnni.cc                     # u8×s8→s32 × 8
├── gemm_f16.cc / gemm_bf16.cc       # f16/bf16 → f32 × 8
├── gemm_mx.cc                       # MX 同格式组合 × Row/Col C
└── gemm_sme_f32.cc                  # SME f32 × 8（非 SME 编译时为空）
examples/  00_basic_gemm.cc  01_benchmark.cc
test/      test_simd.cc  test_gemm.cc  test_gemm_int8.cc  test_gemm_f16.cc
           test_gemm_mx.cc  test_fusion.cc
```
