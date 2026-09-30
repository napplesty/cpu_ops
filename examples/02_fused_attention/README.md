# 02 — Fused Attention (GQA / MHA / MQA)

Assembles the library's primitives — mma policies, panel packing, the
register atom, and the thread pool — into a fused attention kernel
(CUTLASS-style: the composition lives in the example, not in the core
library, exactly like CUTLASS's own FMHA in `examples/41_fused_multi_head_attention`).

**Files**

- `attention_kernel.h` — the fused block kernel: one query panel (≤48 rows)
  per task; S = Q·Kᵀ lands in an L1-sized buffer, online softmax turns it
  into P in place, and O += P·V consumes it before the next kv block loads.
  The Q/K reduction depth (`dk`) and the V/O width (`dv`) are independent.
- `attention.h` — the `Attention<T>` device entry: strided per-(b, h, seq)
  layout, causal block skipping, optional kv-split for decode with a
  deterministic ordered merge. Bit-identical across thread counts with
  kv-split off; deterministic across runs with it on.
- `fused_attention.cc` — self-verifying main (GQA/MHA/MQA × causal/full ×
  chunked prefill × odd shapes × f32/f16/bf16 against a double reference,
  determinism contracts, error paths); `--bench` adds prefill/decode tables.

**Composition seams for sparse/linear variants** (CSA / KDA / DSA): the kv
loop iterates contiguous `[kv0, kv1)` ranges (→ gathered block lists), block
admission is one causal predicate (→ block-diamond masks), and the weight
transform `P = exp2(x − m)` is one function (→ sigmoid-decay for KDA). See
the header comments.

**Build & run**

```bash
cmake --build build --target fused_attention
./build/examples/02_fused_attention/fused_attention           # self-check
./build/examples/02_fused_attention/fused_attention --bench   # + benchmark
```
