// Explicit instantiations of the MLA entry: f32 storage plus the two narrow
// variants (f16/bf16 cache and operands, absorb GEMMs widened to f32).

#include "cpu_ops/attention/mla.h"

namespace cpu_ops {
namespace attention {

template struct MlaAttention<float>;
template struct MlaAttention<float16_t>;
template struct MlaAttention<bfloat16_t>;

}  // namespace attention
}  // namespace cpu_ops
