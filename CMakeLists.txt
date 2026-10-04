cmake_minimum_required(VERSION 3.16)
project(cpu_ops VERSION 0.1.0 LANGUAGES CXX)

option(CPU_OPS_BUILD_EXAMPLES "Build examples" ON)
option(CPU_OPS_BUILD_TESTS "Build tests" ON)
option(CPU_OPS_ENABLE_NATIVE "Compile with -march=native" ON)
option(CPU_OPS_ISA_COMPILE_CHECK
       "Compile-check ISA-gated kernels (AVX-512/AMX) for non-native x86 targets" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

# Header-only template library: the target carries the include path and the
# thread dependency; every kernel is instantiated in the consumer's own
# translation unit, so consumers should also pass the ISA flags matching
# their target (see cpu_ops_build_flags below for the in-repo defaults).
add_library(cpu_ops INTERFACE)
target_include_directories(cpu_ops INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cpu_ops INTERFACE Threads::Threads)
target_compile_features(cpu_ops INTERFACE cxx_std_17)

# Optimization flags shared by the examples and tests.
add_library(cpu_ops_build_flags INTERFACE)
target_compile_options(cpu_ops_build_flags INTERFACE -O3 -Wall -Wextra)
if(CPU_OPS_ENABLE_NATIVE)
  target_compile_options(cpu_ops_build_flags INTERFACE -march=native)
endif()

# Header self-containment check: every public header must compile as a
# standalone translation unit.
file(GLOB_RECURSE CPU_OPS_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include
     ${CMAKE_CURRENT_SOURCE_DIR}/include/cpu_ops/*.h)
set(CPU_OPS_HEADER_CHECK_SOURCES)
foreach(header ${CPU_OPS_HEADERS})
  string(REPLACE "/" "_" tu_name ${header})
  set(tu_file ${CMAKE_CURRENT_BINARY_DIR}/header_check/${tu_name}.cc)
  file(WRITE ${tu_file} "#include \"${header}\"\n")
  list(APPEND CPU_OPS_HEADER_CHECK_SOURCES ${tu_file})
endforeach()
add_library(cpu_ops_header_check OBJECT ${CPU_OPS_HEADER_CHECK_SOURCES})
target_link_libraries(cpu_ops_header_check PRIVATE cpu_ops cpu_ops_build_flags)

# Optional compile check for ISA-gated kernels this host cannot run: builds
# the AVX512-BF16 / AMX policies, the 512-bit MX decode and the MX-VNNI
# policy for recent x86 targets to keep those headers compile-clean
# (no execution).
if(CPU_OPS_ISA_COMPILE_CHECK)
  include(CheckCXXCompilerFlag)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(isa_tu ${CMAKE_CURRENT_BINARY_DIR}/isa_check/isa_kernels.cc)
    file(WRITE ${isa_tu}
         "#include \"cpu_ops/gemm/device/gemm.h\"\n"
         "#include \"cpu_ops/gemm/device/gemm_mx.h\"\n"
         "#include \"cpu_ops/gemm/threadblock/mma_policy_amx.h\"\n"
         "using namespace cpu_ops;\n"
         "template class gemm::device::Gemm<bfloat16_t, layout::RowMajor, bfloat16_t,\n"
         "    layout::RowMajor, float, layout::RowMajor, float,\n"
         "    epilogue::LinearCombination<float>, mma::Bf16GemmConfig, mma::Bf16Policy>;\n"
         "template class gemm::device::Gemm<uint8_t, layout::RowMajor, int8_t,\n"
         "    layout::RowMajor, int32_t, layout::RowMajor, int32_t,\n"
         "    epilogue::LinearCombination<int32_t>, mma::AmxGemmConfig, mma::AmxPolicy>;\n"
         "template class gemm::device::GemmMx<fp8e4m3_t, fp8e4m3_t, layout::RowMajor,\n"
         "    epilogue::LinearCombination<float>, mma::Vnni512GemmConfig,\n"
         "    mma::MxVnniPolicy<fp8e4m3_t, fp8e4m3_t>>;\n"
         "template class gemm::device::GemmMx<fp4e2m1_t, fp4e2m1_t, layout::RowMajor,\n"
         "    epilogue::LinearCombination<float>, mma::Vnni512GemmConfig,\n"
         "    mma::MxVnniPolicy<fp4e2m1_t, fp4e2m1_t>>;\n")
    foreach(isa sapphirerapids icelake-server)
      string(REPLACE "-" "_" isa_id ${isa})
      check_cxx_compiler_flag(-march=${isa} CPU_OPS_HAS_MARCH_${isa_id})
      if(CPU_OPS_HAS_MARCH_${isa_id})
        add_library(cpu_ops_isa_check_${isa_id} OBJECT ${isa_tu})
        target_link_libraries(cpu_ops_isa_check_${isa_id} PRIVATE cpu_ops)
        target_compile_options(cpu_ops_isa_check_${isa_id} PRIVATE -O2 -Wall
                               -march=${isa})
      else()
        message(STATUS "ISA compile check: compiler does not know -march=${isa}, skipped")
      endif()
    endforeach()
  else()
    message(WARNING "CPU_OPS_ISA_COMPILE_CHECK is only meaningful on x86-64; skipping")
  endif()
endif()

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

  add_executable(test_ops test/test_ops.cc)
  target_link_libraries(test_ops PRIVATE cpu_ops cpu_ops_build_flags)

  # Same suites against the portable scalar fallback (no SIMD intrinsics):
  # the headers are compiled again with CPU_OPS_FORCE_SCALAR.
  function(add_scalar_test name test_file)
    add_executable(${name} ${test_file})
    target_compile_definitions(${name} PRIVATE CPU_OPS_FORCE_SCALAR)
    target_link_libraries(${name} PRIVATE cpu_ops cpu_ops_build_flags)
  endfunction()

  add_scalar_test(test_gemm_scalar test/test_gemm.cc)
  add_scalar_test(test_fusion_scalar test/test_fusion.cc)
  add_scalar_test(test_gemm_int8_scalar test/test_gemm_int8.cc)
  add_scalar_test(test_gemm_f16_scalar test/test_gemm_f16.cc)
  add_scalar_test(test_gemm_mx_scalar test/test_gemm_mx.cc)
  add_scalar_test(test_ops_scalar test/test_ops.cc)
  add_test(NAME simd COMMAND test_simd)
  add_test(NAME gemm COMMAND test_gemm)
  add_test(NAME gemm_int8 COMMAND test_gemm_int8)
  add_test(NAME gemm_f16 COMMAND test_gemm_f16)
  add_test(NAME gemm_mx COMMAND test_gemm_mx)
  add_test(NAME ops COMMAND test_ops)
  add_test(NAME fusion COMMAND test_fusion)
  add_test(NAME fusion_scalar COMMAND test_fusion_scalar)
  add_test(NAME gemm_scalar COMMAND test_gemm_scalar)
  add_test(NAME gemm_int8_scalar COMMAND test_gemm_int8_scalar)
  add_test(NAME gemm_f16_scalar COMMAND test_gemm_f16_scalar)
  add_test(NAME gemm_mx_scalar COMMAND test_gemm_mx_scalar)
  add_test(NAME ops_scalar COMMAND test_ops_scalar)
endif()

# Assembled-operator examples (CUTLASS style): each directory owns its
# headers, its self-verifying main, and this CMakeLists; they register with
# CTest when tests are enabled. Kept last so enable_testing() is already done.
if(CPU_OPS_BUILD_EXAMPLES)
  add_subdirectory(examples/02_fused_attention)
  add_subdirectory(examples/03_mla)
  add_subdirectory(examples/04_dsa)
  add_subdirectory(examples/05_csa)
  add_subdirectory(examples/06_kda)
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/cpu_ops-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cpu_ops-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpu_ops)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS cpu_ops EXPORT cpu_ops-targets)
install(EXPORT cpu_ops-targets
  NAMESPACE cpu_ops::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpu_ops)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cpu_ops-config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpu_ops)
