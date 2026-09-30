# Example: DeepSeek Sparse Attention (DSA, DeepSeek-V3.2) — lightning
# indexer + exact top-k + CLS token + no-pack sparse row kernel, over the
# MLA latent cache. Self-verifying; --bench adds the benchmark tables
# (decode vs dense MLA). Depends on ../03_mla for the dense-MLA baseline
# in the benchmark only.

add_executable(dsa dsa.cc)
target_link_libraries(dsa PRIVATE cpu_ops cpu_ops_build_flags)

if(CPU_OPS_BUILD_TESTS)
  # CPU_OPS_SOURCES is relative to the project root; make it absolute here.
  set(LIB_SOURCES_ABS ${CPU_OPS_SOURCES})
  list(TRANSFORM LIB_SOURCES_ABS PREPEND ${PROJECT_SOURCE_DIR}/)
  add_executable(dsa_scalar dsa.cc ${LIB_SOURCES_ABS})
  target_include_directories(dsa_scalar PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(dsa_scalar PRIVATE CPU_OPS_FORCE_SCALAR)
  target_link_libraries(dsa_scalar PRIVATE cpu_ops_build_flags Threads::Threads)

  add_test(NAME dsa COMMAND dsa)
  add_test(NAME dsa_scalar COMMAND dsa_scalar)
endif()
