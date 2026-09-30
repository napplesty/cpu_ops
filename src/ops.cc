// Explicit instantiations for the ops layer (activation / norm); topk is a
// non-template f32 function defined in its header (inline linkage).

#include "cpu_ops/ops/activation.h"
#include "cpu_ops/ops/norm.h"
#include "cpu_ops/ops/topk.h"

namespace cpu_ops {
namespace ops {

#define CPU_OPS_INSTANTIATE(T)                                           \
  template Status gelu<T>(const T*, T*, std::size_t, int);               \
  template Status silu<T>(const T*, T*, std::size_t, int);              \
  template Status rmsnorm<T>(const T*, const T*, T*, int, int, float, int); \
  template Status fused_add_rmsnorm<T>(const T*, const T*, const T*, T*, T*, \
                                       int, int, float, int);            \
  template Status layernorm<T>(const T*, const T*, T*, int, int, float, int);

CPU_OPS_INSTANTIATE(float)
CPU_OPS_INSTANTIATE(cpu_ops::float16_t)
CPU_OPS_INSTANTIATE(cpu_ops::bfloat16_t)

#undef CPU_OPS_INSTANTIATE

}  // namespace ops
}  // namespace cpu_ops
