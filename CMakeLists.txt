cmake_minimum_required(VERSION 3.16)
project(cpu_ops VERSION 0.1.0 LANGUAGES CXX)

option(CPU_OPS_BUILD_EXAMPLES "Build examples" ON)
option(CPU_OPS_BUILD_TESTS "Build tests" ON)
option(CPU_OPS_ENABLE_NATIVE "Compile with -march=native" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(CPU_OPS_SOURCES
  src/gemm_f32.cc
  src/gemm_f64.cc
  src/gemm_f16.cc
  src/gemm_bf16.cc
  src/gemm_vnni.cc
  src/gemm_mx.cc
  src/gemm_sme_f32.cc
  src/thread_pool.cc
  src/thread_swizzle.cc)

add_library(cpu_ops STATIC ${CPU_OPS_SOURCES})
target_include_directories(cpu_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cpu_ops PUBLIC Threads::Threads)

# Optimization flags shared by the library, examples and tests.
add_library(cpu_ops_build_flags INTERFACE)
target_compile_options(cpu_ops_build_flags INTERFACE -O3 -Wall -Wextra)
if(CPU_OPS_ENABLE_NATIVE)
  target_compile_options(cpu_ops_build_flags INTERFACE -march=native)
endif()

target_link_libraries(cpu_ops PRIVATE cpu_ops_build_flags)

if(CPU_OPS_BUILD_EXAMPLES)
  add_executable(basic_gemm examples/00_basic_gemm.cc)
  target_link_libraries(basic_gemm PRIVATE cpu_ops cpu_ops_build_flags)

  add_executable(benchmark examples/01_benchmark.cc)
  target_link_libraries(benchmark PRIVATE cpu_ops cpu_ops_build_flags)
endif()

if(CPU_OPS_BUILD_TESTS)
  enable_testing()

  add_executable(test_simd test/test_simd.cc)
  target_link_libraries(test_simd PRIVATE cpu_ops cpu_ops_build_flags)

  add_executable(test_gemm test/test_gemm.cc)
  target_link_libraries(test_gemm PRIVATE cpu_ops cpu_ops_build_flags)

  add_executable(test_gemm_int8 test/test_gemm_int8.cc)
  target_link_libraries(test_gemm_int8 PRIVATE cpu_ops cpu_ops_build_flags)

  add_executable(test_gemm_f16 test/test_gemm_f16.cc)
  target_link_libraries(test_gemm_f16 PRIVATE cpu_ops cpu_ops_build_flags)

  add_executable(test_gemm_mx test/test_gemm_mx.cc)
  target_link_libraries(test_gemm_mx PRIVATE cpu_ops cpu_ops_build_flags)

  add_executable(test_fusion test/test_fusion.cc)
  target_link_libraries(test_fusion PRIVATE cpu_ops cpu_ops_build_flags)

  # Same suites against the portable scalar fallback (no SIMD intrinsics):
  # each pairs a test file with the library sources compiled with
  # CPU_OPS_FORCE_SCALAR.
  function(add_scalar_test name test_file)
    add_executable(${name} ${test_file} ${CPU_OPS_SOURCES})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(${name} PRIVATE CPU_OPS_FORCE_SCALAR)
    target_link_libraries(${name} PRIVATE cpu_ops_build_flags Threads::Threads)
  endfunction()

  add_scalar_test(test_gemm_scalar test/test_gemm.cc)
  add_scalar_test(test_fusion_scalar test/test_fusion.cc)
  add_scalar_test(test_gemm_int8_scalar test/test_gemm_int8.cc)
  add_scalar_test(test_gemm_f16_scalar test/test_gemm_f16.cc)
  add_scalar_test(test_gemm_mx_scalar test/test_gemm_mx.cc)

  add_test(NAME simd COMMAND test_simd)
  add_test(NAME gemm COMMAND test_gemm)
  add_test(NAME gemm_int8 COMMAND test_gemm_int8)
  add_test(NAME gemm_f16 COMMAND test_gemm_f16)
  add_test(NAME gemm_mx COMMAND test_gemm_mx)
  add_test(NAME fusion COMMAND test_fusion)
  add_test(NAME fusion_scalar COMMAND test_fusion_scalar)
  add_test(NAME gemm_scalar COMMAND test_gemm_scalar)
  add_test(NAME gemm_int8_scalar COMMAND test_gemm_int8_scalar)
  add_test(NAME gemm_f16_scalar COMMAND test_gemm_f16_scalar)
  add_test(NAME gemm_mx_scalar COMMAND test_gemm_mx_scalar)
endif()
