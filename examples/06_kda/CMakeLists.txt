# Example: KDA (Kimi Delta Attention) — gated delta-rule linear attention
# with naive (decode) and chunkwise (prefill/training) execution modes.
# Self-verifying; --bench adds naive-vs-chunked benchmark numbers.
# Depends on ../04_dsa for the streaming row-kernel primitives.

add_executable(kda kda.cc)
target_link_libraries(kda PRIVATE cpu_ops cpu_ops_build_flags)

if(CPU_OPS_BUILD_TESTS)
  # CPU_OPS_SOURCES is relative to the project root; make it absolute here.
  set(LIB_SOURCES_ABS ${CPU_OPS_SOURCES})
  list(TRANSFORM LIB_SOURCES_ABS PREPEND ${PROJECT_SOURCE_DIR}/)
  add_executable(kda_scalar kda.cc ${LIB_SOURCES_ABS})
  target_include_directories(kda_scalar PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(kda_scalar PRIVATE CPU_OPS_FORCE_SCALAR)
  target_link_libraries(kda_scalar PRIVATE cpu_ops_build_flags Threads::Threads)

  add_test(NAME kda COMMAND kda)
  add_test(NAME kda_scalar COMMAND kda_scalar)
endif()
