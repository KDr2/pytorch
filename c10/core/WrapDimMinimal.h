#pragma once

#include <c10/core/SymInt.h>
#include <c10/util/Exception.h>

namespace c10 {

namespace detail {
template <typename T>
C10_API T maybe_wrap_dim_slow(T dim, T dim_post_expr, bool wrap_scalar);
} // namespace detail

static inline int64_t maybe_wrap_dim(
    int64_t dim,
    int64_t dim_post_expr,
    bool wrap_scalar = true) {
  // Inline the fast paths
  if (C10_LIKELY(dim_post_expr * -1 <= dim && dim < dim_post_expr)) {
    // Branch-less version of dim + (dim < 0 ? dim_post_expr : 0)
    return dim + dim_post_expr * (dim < 0);
  }
  // Check edge-cases out-of-line (wrapping scalars and out-of-bounds errors)
  return c10::detail::maybe_wrap_dim_slow<int64_t>(
      dim, dim_post_expr, wrap_scalar);
}

static inline SymInt maybe_wrap_dim(
    SymInt dim,
    SymInt dim_post_expr,
    bool wrap_scalar = true) {
  // Inline the fast paths
  if (C10_LIKELY(dim_post_expr * -1 <= dim && dim < dim_post_expr)) {
    // Branch-less version of dim + (dim < 0 ? dim_post_expr : 0)
    return dim + dim_post_expr * (dim < 0);
  }
  // Check edge-cases out-of-line (wrapping scalars and out-of-bounds errors)
  return c10::detail::maybe_wrap_dim_slow<SymInt>(
      dim, dim_post_expr, wrap_scalar);
}

} // namespace c10
