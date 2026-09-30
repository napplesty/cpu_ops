// Explicit instantiations of the fused attention entry: f32 storage plus the
// two narrow-operand variants (f16/bf16 widened to f32 at pack time).

#include "cpu_ops/attention/attention.h"

namespace cpu_ops {
namespace attention {

template struct Attention<float>;
template struct Attention<float16_t>;
template struct Attention<bfloat16_t>;

}  // namespace attention
}  // namespace cpu_ops
