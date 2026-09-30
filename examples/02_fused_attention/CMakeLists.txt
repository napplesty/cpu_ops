# Example: fused attention (GQA / MHA / MQA) assembled from cpu_ops
# primitives. Self-verifying (run without arguments); --bench adds the
# benchmark tables. When tests are enabled, the example and its scalar
# fallback build are registered with CTest.

add_executable(fused_attention fused_attention.cc)
target_link_libraries(fused_attention PRIVATE cpu_ops cpu_ops_build_flags)

if(CPU_OPS_BUILD_TESTS)
  # CPU_OPS_SOURCES is relative to the project root; make it absolute here.
  set(LIB_SOURCES_ABS ${CPU_OPS_SOURCES})
  list(TRANSFORM LIB_SOURCES_ABS PREPEND ${PROJECT_SOURCE_DIR}/)
  add_executable(fused_attention_scalar fused_attention.cc ${LIB_SOURCES_ABS})
  target_include_directories(fused_attention_scalar PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(fused_attention_scalar PRIVATE CPU_OPS_FORCE_SCALAR)
  target_link_libraries(fused_attention_scalar PRIVATE cpu_ops_build_flags Threads::Threads)

  add_test(NAME fused_attention COMMAND fused_attention)
  add_test(NAME fused_attention_scalar COMMAND fused_attention_scalar)
endif()
