#pragma once

// cpu_ops: header-only CPU GEMM / ops template library (CUTLASS-style).
//
//   #include <cpu_ops/cpu_ops.h>   // umbrella: the whole public API
//
// Layering (mirroring CUTLASS):
//   arch/              ISA primitives: simd::Vec, mma atoms
//   gemm/threadblock/  mma policies, panel packing, cache-blocked mainloop
//   gemm/kernel/       run_blocked: thread partition + split-k orchestration
//   gemm/device/       device-level operators: Gemm, GemmMx, fused aliases
//   epilogue/          LinearCombination, fused op chains, split-k PartialSum
//   ops/               activation / norm / topk primitives
//   thread/            persistent thread pool

#include "cpu_ops/epilogue/fusion.h"
#include "cpu_ops/epilogue/linear_combination.h"
#include "cpu_ops/gemm/device/gemm.h"
#include "cpu_ops/gemm/device/gemm_mx.h"
#include "cpu_ops/gemm_coord.h"
#include "cpu_ops/layout/matrix.h"
#include "cpu_ops/matrix_shape.h"
#include "cpu_ops/mx_formats.h"
#include "cpu_ops/numeric_types.h"
#include "cpu_ops/ops/activation.h"
#include "cpu_ops/ops/norm.h"
#include "cpu_ops/ops/topk.h"
#include "cpu_ops/status.h"
#include "cpu_ops/tensor_ref.h"
