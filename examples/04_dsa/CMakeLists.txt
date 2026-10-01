# Example: DeepSeek Sparse Attention (DSA, DeepSeek-V3.2) — lightning
# indexer + exact top-k + CLS token + no-pack sparse row kernel, over the
# MLA latent cache. Self-verifying; --bench adds the benchmark tables
# (decode vs dense MLA). Depends on ../03_mla for the dense-MLA baseline
# in the benchmark only.

add_executable(dsa dsa.cc)
target_link_libraries(dsa PRIVATE cpu_ops cpu_ops_build_flags)

if(CPU_OPS_BUILD_TESTS)
  # Scalar fallback variant: same sources, headers compiled with
  # CPU_OPS_FORCE_SCALAR.
  add_executable(dsa_scalar dsa.cc)
  target_compile_definitions(dsa_scalar PRIVATE CPU_OPS_FORCE_SCALAR)
  target_link_libraries(dsa_scalar PRIVATE cpu_ops cpu_ops_build_flags)

  add_test(NAME dsa COMMAND dsa)
  add_test(NAME dsa_scalar COMMAND dsa_scalar)
endif()
