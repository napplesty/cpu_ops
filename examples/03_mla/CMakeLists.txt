# Example: MLA (DeepSeek-V2/V3, weight-absorbed) composed from the fused
# attention example plus the library's GEMM. Self-verifying; --bench adds
# the benchmark tables. Depends on ../02_fused_attention for the core.

add_executable(mla mla.cc)
target_link_libraries(mla PRIVATE cpu_ops cpu_ops_build_flags)

if(CPU_OPS_BUILD_TESTS)
  # CPU_OPS_SOURCES is relative to the project root; make it absolute here.
  set(LIB_SOURCES_ABS ${CPU_OPS_SOURCES})
  list(TRANSFORM LIB_SOURCES_ABS PREPEND ${PROJECT_SOURCE_DIR}/)
  add_executable(mla_scalar mla.cc ${LIB_SOURCES_ABS})
  target_include_directories(mla_scalar PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(mla_scalar PRIVATE CPU_OPS_FORCE_SCALAR)
  target_link_libraries(mla_scalar PRIVATE cpu_ops_build_flags Threads::Threads)

  add_test(NAME mla COMMAND mla)
  add_test(NAME mla_scalar COMMAND mla_scalar)
endif()
