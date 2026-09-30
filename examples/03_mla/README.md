# 03 — MLA (Multi-head Latent Attention, DeepSeek-V2/V3)

Weight-absorbed MLA on top of the fused attention example (`mla.h` includes
`../02_fused_attention/attention.h`) and the library's pre-instantiated GEMM
— an assembly example in the CUTLASS sense: nothing here is linked into
`libcpu_ops.a`.

**Shape of the composition**

1. per query head: absorb GEMM `q̃ = q_nope · W_UK_hᵀ` (`[sq×dn] × [dn×dc]`),
2. `q_pe` appended after `q̃` (the query becomes `[sq × (dc+dp)]`),
3. one fused attention call over the shared latent cache: `heads_kv = 1`,
   `dim = dc + dp` (576 in DeepSeek), `dim_v = dc` — K and V are the *same*
   buffer with different widths,
4. per query head: un-absorb GEMM `o = õ · W_UV_hᵀ` (`[sq×dc] × [dc×dn]`).

The self-check verifies against a *non-absorbed* double reference
(`k_nope = W_UK·c`, `v = W_UV·c`), which also validates the absorption
identity and the GEMM orientations.

**Cache**: one shared `[seq_kv × (dc + dp)]` row per token (bf16 → 1152
B/token for DeepSeek shapes) instead of per-head K/V. `q_pe`/`k_pe` must be
RoPE-rotated upstream; the softmax scale is `1/sqrt(dn + dp)`.

**Build & run**

```bash
cmake --build build --target mla
./build/examples/03_mla/mla           # self-check
./build/examples/03_mla/mla --bench   # + benchmark
```
