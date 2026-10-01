# Example: Compressed Sparse Attention (CSA, NSA-style: compressed +
# selected + sliding-window branches with sigmoid gates) over the MLA
# latent cache. Self-verifying; --bench adds reference benchmark numbers.
# Depends on ../04_dsa for the streaming row-kernel primitives.

add_executable(csa csa.cc)
target_link_libraries(csa PRIVATE cpu_ops cpu_ops_build_flags)

if(CPU_OPS_BUILD_TESTS)
  # Scalar fallback variant: same sources, headers compiled with
  # CPU_OPS_FORCE_SCALAR.
  add_executable(csa_scalar csa.cc)
  target_compile_definitions(csa_scalar PRIVATE CPU_OPS_FORCE_SCALAR)
  target_link_libraries(csa_scalar PRIVATE cpu_ops cpu_ops_build_flags)

  add_test(NAME csa COMMAND csa)
  add_test(NAME csa_scalar COMMAND csa_scalar)
endif()
