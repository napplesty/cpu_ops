#pragma once

namespace cpu_ops {

enum class Status {
  kSuccess = 0,
  kErrorInvalidProblem,    // negative or otherwise malformed problem size
  kErrorInvalidArguments,  // null pointer or insufficient leading dimension
};

}  // namespace cpu_ops
