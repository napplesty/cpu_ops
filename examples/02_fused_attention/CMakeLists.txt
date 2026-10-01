# Example: fused attention (GQA / MHA / MQA) assembled from cpu_ops
# primitives. Self-verifying (run without arguments); --bench adds the
# benchmark tables. When tests are enabled, the example and its scalar
# fallback build are registered with CTest.

add_executable(fused_attention fused_attention.cc)
target_link_libraries(fused_attention PRIVATE cpu_ops cpu_ops_build_flags)

if(CPU_OPS_BUILD_TESTS)
  # Scalar fallback variant: same sources, headers compiled with
  # CPU_OPS_FORCE_SCALAR.
  add_executable(fused_attention_scalar fused_attention.cc)
  target_compile_definitions(fused_attention_scalar PRIVATE CPU_OPS_FORCE_SCALAR)
  target_link_libraries(fused_attention_scalar PRIVATE cpu_ops cpu_ops_build_flags)

  add_test(NAME fused_attention COMMAND fused_attention)
  add_test(NAME fused_attention_scalar COMMAND fused_attention_scalar)
endif()
