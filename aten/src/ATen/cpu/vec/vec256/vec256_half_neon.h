#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/Half.h>
#include <c10/util/irange.h>

using c10::Half;

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// Right now contains only aarch64 implementation.
// Due to follow two reasons aarch32 is not currently supported.
// 1. Due to difference in ISA been aarch32 and aarch64, intrinsics
//    that work for aarch64 dont work for aarch32.
// 2. Android NDK r21 has problems with compiling aarch32.
//    Clang seg faults.
//    https://github.com/android/ndk/issues/1248
//    https://bugs.llvm.org/show_bug.cgi?id=45824
// Most likely we will do aarch32 support with inline asm.
#if defined(__aarch64__)

#ifdef __BIG_ENDIAN__
#error "Big endian is not supported."
#endif

template <int index, bool mask_val>
struct BlendHalfRegs {
  static float16x8_t impl(
      const float16x8_t& a,
      const float16x8_t& b,
      float16x8_t& res);
};

template <int index>
struct BlendHalfRegs<index, true> {
  static float16x8_t impl(
      const float16x8_t& a,
      const float16x8_t& b,
      float16x8_t& res) {
    return vsetq_lane_f16(vgetq_lane_f16(b, index), res, index);
  }
};

template <int index>
struct BlendHalfRegs<index, false> {
  static float16x8_t impl(
      const float16x8_t& a,
      const float16x8_t& b,
      float16x8_t& res) {
    return vsetq_lane_f16(vgetq_lane_f16(a, index), res, index);
  }
};

// On ARM, Half type supports float16_t->Half constructor and Half->float16_t
// conversion
template <>
class Vectorized<Half> {
 private:
  float16x8x2_t values;

 public:
  using value_type = float16_t;
  using size_type = int;
  static constexpr size_type size() {
    return 16;
  }
  private:
  // Several math functions have no Half overloads.  We use map_float to handle
  // these.
  Vectorized<Half> map_float(float (*const f)(float)) const {
    __at_align__ Half tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  // Several math functions take 2 arguments.  We use map2 to handle these.
  // The first argument passed to the map function f is always self.
  Vectorized<Half> map2(
      const Vectorized<Half>& second,
      float16_t (*const f)(float16_t, float16_t)) const {
    __at_align__ Half tmp_first[size()];
    __at_align__ Half tmp_second[size()];
    store(tmp_first); // store this to tmp_first
    second.store(tmp_second);
    for (const auto i : c10::irange(size())) {
      tmp_first[i] = f(tmp_first[i], tmp_second[i]);
    }
    return loadu(tmp_first);
  }

  Vectorized<Half> map2_float(
      const Vectorized<Half>& second,
      float (*const f)(float, float)) const {
    __at_align__ float16_t tmp_first[size()];
    __at_align__ float16_t tmp_second[size()];
    store(tmp_first); // store this to tmp_first
    second.store(tmp_second);
    for (const auto i : c10::irange(size())) {
      tmp_first[i] = f(tmp_first[i], tmp_second[i]);
    }
    return loadu(tmp_first);
  }

  public:
  Vectorized() {}
  Vectorized(float16x8x2_t v) : values(v) {}

  Vectorized(float16_t val) : values{vdupq_n_f16(val), vdupq_n_f16(val)} {}
  Vectorized(
      float16_t val0,
      float16_t val1,
      float16_t val2,
      float16_t val3,
      float16_t val4,
      float16_t val5,
      float16_t val6,
      float16_t val7,
      float16_t val8,
      float16_t val9,
      float16_t val10,
      float16_t val11,
      float16_t val12,
      float16_t val13,
      float16_t val14,
      float16_t val15)
      : values{
            val0,
            val1,
            val2,
            val3,
            val4,
            val5,
            val6,
            val7,
            val8,
            val9,
            val10,
            val11,
            val12,
            val13,
            val14,
            val15} {}
  Vectorized(float16x8_t val0, float16x8_t val1) : values{val0, val1} {}
  operator float16x8x2_t() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<Half> blend(
      const Vectorized<Half>& a,
      const Vectorized<Half>& b) {
    Vectorized<Half> vec;
    // 0.
    vec.values.val[0] = BlendHalfRegs<0, (mask & 0x01) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<1, (mask & 0x02) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<2, (mask & 0x04) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<3, (mask & 0x08) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);

    vec.values.val[0] = BlendHalfRegs<4, (mask & 0x10) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<5, (mask & 0x20) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<6, (mask & 0x40) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<7, (mask & 0x80) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);

    // 1.
    vec.values.val[1] = BlendHalfRegs<0, (mask & 0x10) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<1, (mask & 0x20) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<2, (mask & 0x40) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<3, (mask & 0x80) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);

    vec.values.val[1] = BlendHalfRegs<4, (mask & 0x10) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<5, (mask & 0x20) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<6, (mask & 0x40) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<7, (mask & 0x80) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);

    return vec;
  }
  static Vectorized<Half> blendv(
      const Vectorized<Half>& a,
      const Vectorized<Half>& b,
      const Vectorized<Half>& mask) {
    // Note: using blendv is very awkward because 0xFFFF is one of many NaN's in
    // FP16 It's unfortunate that the mask has type Half (required from
    // vec_base)

    // TODO
    // NB: This requires that each value, i.e., each uint value,
    // of the mask either all be zeros or all be 1s.
    // We perhaps need some kind of an assert?
    // But that will affect performance.
    Vectorized<Half> vec(mask.values);
    vec.values.val[0] = vbslq_f16(
        vreinterpretq_u16_f16(vec.values.val[0]),
        b.values.val[0],
        a.values.val[0]);
    vec.values.val[1] = vbslq_f16(
        vreinterpretq_u16_f16(vec.values.val[1]),
        b.values.val[1],
        a.values.val[1]);
    return vec;
  }
  template <typename step_t>
  static Vectorized<Half> arange(
      Half base = 0.0,
      step_t step = static_cast<step_t>(1)) {
    const Vectorized<Half> base_vec(base);
    const Vectorized<Half> step_vec(step);
    const Vectorized<Half> step_sizes(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    return fmadd(step_sizes, step_vec, base_vec);
  }
  static Vectorized<Half> set(
      const Vectorized<Half>& a,
      const Vectorized<Half>& b,
      int64_t count = size()) {
    uint16_t pre_mask[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < count; i++) {
      pre_mask[i] = 0xFFFF;
    }
    uint16x8x2_t mask = vld1q_u16_x2(pre_mask);

    // Using blendv is awkward because 0xFFFF is one of many NaN's in FP16
    // so we directly use vbslq_f16 instead
    Vectorized<Half> vec(
        vbslq_f16(
            // Low bits
            mask.val[0],
            b.values.val[0],
            a.values.val[0]),
        // High bits
        vbslq_f16(mask.val[1], b.values.val[1], a.values.val[1]));

    return vec;
  }
  static Vectorized<Half> loadu(const void* ptr, int64_t count = size()) {
    if (count == size()) {
      return vld1q_f16_x2(reinterpret_cast<const float16_t*>(ptr));
    } else if (count == (size() >> 1)) {
      Vectorized<Half> res;
      res.values.val[0] = vld1q_f16(reinterpret_cast<const float16_t*>(ptr));
      res.values.val[1] = vdupq_n_f16(0);
      return res;
    } else {
      __at_align__ float16_t tmp_values[size()];
      for (const auto i : c10::irange(size())) {
        tmp_values[i] = 0;
      }
      std::memcpy(
          tmp_values,
          reinterpret_cast<const float16_t*>(ptr),
          count * sizeof(float16_t));
      return vld1q_f16_x2(reinterpret_cast<const float16_t*>(tmp_values));
    }
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      vst1q_f16_x2(reinterpret_cast<float16_t*>(ptr), values);
      return;
    } else if (count == (size() >> 1)) {
      vst1q_f16(reinterpret_cast<float16_t*>(ptr), values.val[0]);
    } else {
      float16_t tmp_values[size()];
      vst1q_f16_x2(reinterpret_cast<float16_t*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float16_t));
    }
  }
  inline const float16x8_t& get_low() const {
    return values.val[0];
  }
  inline float16x8_t& get_low() {
    return values.val[0];
  }
  inline const float16x8_t& get_high() const {
    return values.val[1];
  }
  inline float16x8_t& get_high() {
    return values.val[1];
  }
  // Very slow implementation of indexing.
  // Only required because vec256_qint refers to this.
  // Once we specialize that implementation for ARM
  // this should be removed. TODO (kimishpatel)
  Half operator[](int idx) const {
    __at_align__ Half tmp[size()];
    store(tmp);
    return tmp[idx];
  }
  Half operator[](int idx) {
    __at_align__ Half tmp[size()];
    store(tmp);
    return tmp[idx];
  }
  // For boolean version where we want to if any 1/all zero
  // etc. can be done faster in a different way.
  int zero_mask() const {
    __at_align__ Half tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (tmp[i] == 0) {
        mask |= (1 << i);
      }
    }
    return mask;
  }
  Vectorized<Half> isnan() const {
    __at_align__ Half tmp[size()];
    __at_align__ Half res[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if (_isnan(tmp[i])) {
        std::memset(static_cast<void*>(&res[i]), 0xFF, sizeof(Half));
      } else {
        std::memset(static_cast<void*>(&res[i]), 0, sizeof(Half));
      }
    }
    return loadu(res);
  };
  bool has_inf_nan() const {
    __at_align__ Half tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if (_isnan(tmp[i]) ||
          _isinf(tmp[i])) {
        return true;
      }
    }
    return false;
  }
  Vectorized<Half> map(Half (*const f)(Half)) const {
    __at_align__ Half tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<Half> abs() const {
    return Vectorized<Half>(vabsq_f16(values.val[0]), vabsq_f16(values.val[1]));
  }
  Vectorized<Half> angle() const {
    auto zero = Vectorized<Half>(0);
    auto pi = Vectorized<Half>(c10::pi<Half>);
    auto tmp = blendv(zero, pi, *this < zero);
    return blendv(tmp, *this, isnan());
  }
  Vectorized<Half> real() const {
    return *this;
  }
  Vectorized<Half> imag() const {
    return Vectorized<Half>(0);
  }
  Vectorized<Half> conj() const {
    return *this;
  }

  // Sleef does not support FP16.
  // The following math functions are implemented with map
  // In future, if performance testing shows it's needed, we could try applying
  // the FP32 Sleef functions on 4 Halfs at a time, but for now we go with map
  // solution
  Vectorized<Half> acos() const {
    return map(std::acos);
  }
  Vectorized<Half> acosh() const {
    return map_float(std::acosh);
  }
  Vectorized<Half> asin() const {
    return map(std::asin);
  }
  Vectorized<Half> atan() const {
    return map(std::atan);
  }
  Vectorized<Half> atanh() const {
    return map(std::atanh);
  }
  Vectorized<Half> atan2(const Vectorized<Half>& exp) const {
    return map2_float(exp, std::atan2);
  }
  Vectorized<Half> copysign(const Vectorized<Half>& sign) const {
    return map2_float(sign, std::copysign);
  }
  Vectorized<Half> erf() const {
    return map(std::erf);
  }
  Vectorized<Half> erfc() const {
    return map(std::erfc);
  }
  Vectorized<Half> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<Half> exp() const {
    return map(std::exp);
  }
  Vectorized<Half> exp2() const {
    return map_float(std::exp2);
  }
  Vectorized<Half> expm1() const {
    return map(std::expm1);
  }
  Vectorized<Half> exp_u20() const {
    return exp();
  }
  Vectorized<Half> fmod(const Vectorized<Half>& q) const {
    return map2(q, std::fmod);
  }
  Vectorized<Half> hypot(const Vectorized<Half>& b) const {
    return map2_float(b, std::hypot);
  }
  Vectorized<Half> i0() const {
    return map_float(calc_i0);
  }
  Vectorized<Half> i0e() const {
    return map(calc_i0e);
  }
  Vectorized<Half> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<Half> igamma(const Vectorized<Half>& x) const {
    return map2(x, calc_igamma);
  }
  Vectorized<Half> igammac(const Vectorized<Half>& x) const {
    return map2(x, calc_igammac);
  }
  Vectorized<Half> log() const {
    return map(std::log);
  }
  Vectorized<Half> log10() const {
    return map(std::log10);
  }
  Vectorized<Half> log1p() const {
    return map(std::log1p);
  }
  Vectorized<Half> log2() const {
    return map(std::log2);
  }
  Vectorized<Half> nextafter(const Vectorized<Half>& b) const {
    return map2(b, std::nextafter);
  }
  Vectorized<Half> frac() const;
  Vectorized<Half> sin() const {
    return map(std::sin);
  }
  Vectorized<Half> sinh() const {
    return map(std::sinh);
  }
  Vectorized<Half> cos() const {
    return map(std::cos);
  }
  Vectorized<Half> cosh() const {
    return map(std::cosh);
  }
  Vectorized<Half> ceil() const {
    return map(at::native::ceil_impl);
  }
  Vectorized<Half> floor() const {
    return map(at::native::floor_impl);
  }
  Vectorized<Half> neg() const {
    return Vectorized<Half>(vnegq_f16(values.val[0]), vnegq_f16(values.val[1]));
  }
  inline Vectorized<Half> round() const {
    return map(at::native::round_impl);
  }
  inline Vectorized<Half> tan() const {
    return map(std::tan);
  }
  inline Vectorized<Half> tanh() const {
    return map(std::tanh);
  }
  Vectorized<Half> trunc() const {
    float16x8_t r0 = vrndq_f16(values.val[0]);
    float16x8_t r1 = vrndq_f16(values.val[1]);
    return Vectorized<Half>(r0, r1);
  }
  Vectorized<Half> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<Half> sqrt() const {
    return Vectorized<Half>(
        vsqrtq_f16(values.val[0]), vsqrtq_f16(values.val[1]));
  }
  Vectorized<Half> reciprocal() const {
    auto r0 = vrecpeq_f16(values.val[0]);
    auto r1 = vrecpeq_f16(values.val[1]);
    return Vectorized<Half>(r0, r1);
  }
  Vectorized<Half> rsqrt() const {
    auto r0 = vrsqrteq_f16(values.val[0]);
    auto r1 = vrsqrteq_f16(values.val[1]);
    return Vectorized<Half>(r0, r1);
  }
  Vectorized<Half> pow(const Vectorized<Half>& exp) const {
    return map2(exp, std::pow);
  }
  Vectorized<Half> operator==(const Vectorized<Half>& other) const {
    float16x8_t r0 =
        vreinterpretq_f16_u16(vceqq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vceqq_f16(values.val[1], other.values.val[1]));
    return Vectorized<Half>(r0, r1);
  }

  Vectorized<Half> operator!=(const Vectorized<Half>& other) const {
    float32x4_t r0 = vreinterpretq_f16_u16(
        vmvnq_u16(vceqq_f16(values.val[0], other.values.val[0])));
    float32x4_t r1 = vreinterpretq_f16_u16(
        vmvnq_u16(vceqq_f16(values.val[1], other.values.val[1])));
    return Vectorized<Half>(r0, r1);
  }

  Vectorized<Half> operator<(const Vectorized<Half>& other) const {
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcltq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcltq_f16(values.val[1], other.values.val[1]));
    return Vectorized<Half>(r0, r1);
  }

  Vectorized<Half> operator<=(const Vectorized<Half>& other) const {
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcleq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcleq_f16(values.val[1], other.values.val[1]));
    return Vectorized<Half>(r0, r1);
  }

  Vectorized<Half> operator>(const Vectorized<Half>& other) const {
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcgtq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcgtq_f16(values.val[1], other.values.val[1]));
    return Vectorized<Half>(r0, r1);
  }

  Vectorized<Half> operator>=(const Vectorized<Half>& other) const {
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcgeq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcgeq_f16(values.val[1], other.values.val[1]));
    return Vectorized<Half>(r0, r1);
  }

  Vectorized<Half> eq(const Vectorized<Half>& other) const;
  Vectorized<Half> ne(const Vectorized<Half>& other) const;
  Vectorized<Half> gt(const Vectorized<Half>& other) const;
  Vectorized<Half> ge(const Vectorized<Half>& other) const;
  Vectorized<Half> lt(const Vectorized<Half>& other) const;
  Vectorized<Half> le(const Vectorized<Half>& other) const;
}; // Vectorized<Half>

template <>
Vectorized<Half> inline operator+(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  float16x8_t r0 = vaddq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vaddq_f16(a.get_high(), b.get_high());
#else
   return convert_float_half(convert_to_float(a) + convert_to_float(b));
#endif
  return Vectorized<Half>(r0, r1);
}

template <>
Vectorized<Half> inline operator-(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  float16x8_t r0 = vsubq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vsubq_f16(a.get_high(), b.get_high());
  return Vectorized<Half>(r0, r1);
}

template <>
Vectorized<Half> inline operator*(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  float16x8_t r0 = vmulq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vmulq_f16(a.get_high(), b.get_high());
  return Vectorized<Half>(r0, r1);
}

template <>
Vectorized<Half> inline operator/(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  float16x8_t r0 = vdivq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vdivq_f16(a.get_high(), b.get_high());
  return Vectorized<Half>(r0, r1);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<Half> Vectorized<Half>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<Half> inline maximum(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  float16x8_t r0 = vmaxq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vmaxq_f16(a.get_high(), b.get_high());
  return Vectorized<Half>(r0, r1);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<Half> inline minimum(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  float16x8_t r0 = vminq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vminq_f16(a.get_high(), b.get_high());
  return Vectorized<Half>(r0, r1);
}

template <>
Vectorized<Half> inline clamp(
    const Vectorized<Half>& a,
    const Vectorized<Half>& min,
    const Vectorized<Half>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<Half> inline clamp_max(
    const Vectorized<Half>& a,
    const Vectorized<Half>& max) {
  return minimum(max, a);
}

template <>
Vectorized<Half> inline clamp_min(
    const Vectorized<Half>& a,
    const Vectorized<Half>& min) {
  return maximum(min, a);
}

template <>
Vectorized<Half> inline operator&(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  float16x8_t r0 = vreinterpretq_f16_u16(vandq_u16(
      vreinterpretq_u16_f16(a.get_low()), vreinterpretq_u16_f16(b.get_low())));
  float16x8_t r1 = vreinterpretq_f16_u16(vandq_u16(
      vreinterpretq_u16_f16(a.get_high()),
      vreinterpretq_u16_f16(b.get_high())));
  return Vectorized<Half>(r0, r1);
}

template <>
Vectorized<Half> inline operator|(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  float16x8_t r0 = vreinterpretq_f16_u16(vorrq_u16(
      vreinterpretq_u16_f16(a.get_low()), vreinterpretq_u16_f16(b.get_low())));
  float16x8_t r1 = vreinterpretq_f16_u16(vorrq_u16(
      vreinterpretq_u16_f16(a.get_high()),
      vreinterpretq_u16_f16(b.get_high())));
  return Vectorized<Half>(r0, r1);
}

template <>
Vectorized<Half> inline operator^(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  float16x8_t r0 = vreinterpretq_f16_u16(veorq_u16(
      vreinterpretq_u16_f16(a.get_low()), vreinterpretq_u16_f16(b.get_low())));
  float16x8_t r1 = vreinterpretq_f16_u16(veorq_u16(
      vreinterpretq_u16_f16(a.get_high()),
      vreinterpretq_u16_f16(b.get_high())));
  return Vectorized<Half>(r0, r1);
}

inline Vectorized<Half> Vectorized<Half>::eq(
    const Vectorized<Half>& other) const {
  return (*this == other) &
      Vectorized<Half>(1);
}

inline Vectorized<Half> Vectorized<Half>::ne(
    const Vectorized<Half>& other) const {
  return (*this != other) &
      Vectorized<Half>(1);
}

inline Vectorized<Half> Vectorized<Half>::gt(
    const Vectorized<Half>& other) const {
  return (*this > other) &
      Vectorized<Half>(1);
}

inline Vectorized<Half> Vectorized<Half>::ge(
    const Vectorized<Half>& other) const {
  return (*this >= other) &
      Vectorized<Half>(1);
}

inline Vectorized<Half> Vectorized<Half>::lt(
    const Vectorized<Half>& other) const {
  return (*this < other) &
      Vectorized<Half>(1);
}

inline Vectorized<Half> Vectorized<Half>::le(
    const Vectorized<Half>& other) const {
  return (*this <= other) &
      Vectorized<Half>(1);
}

template <>
inline void convert(const float16_t* src, int16_t* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorized<Half>::size());
       i += Vectorized<Half>::size()) {
    vst1q_s16(dst + i, vcvtq_s16_f16(vld1q_f16(src + i)));
    vst1q_s16(dst + i + 8, vcvtq_s16_f16(vld1q_f16(src + i + 8)));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = static_cast<int16_t>(
        src[i]);
  }
}

template <>
inline void convert(const int16_t* src, float16_t* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorized<Half>::size());
       i += Vectorized<Half>::size()) {
    vst1q_f16(dst + i, vcvtq_f16_s16(vld1q_s16(src + i)));
    vst1q_f16(dst + i + 8, vcvtq_f16_s16(vld1q_s16(src + i + 8)));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = static_cast<float16_t>(
        src[i]);
  }
}

template <>
Vectorized<Half> inline fmadd(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b,
    const Vectorized<Half>& c) {
  float16x8_t r0 = vfmaq_f16(c.get_low(), a.get_low(), b.get_low());
  float16x8_t r1 = vfmaq_f16(c.get_high(), a.get_high(), b.get_high());
  return Vectorized<Half>(r0, r1);
}

template <>
Vectorized<Half> inline fmsub(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b,
    const Vectorized<Half>& c) {
  float16x8_t r0 = vfmsq_f16(c.get_low(), a.get_low(), b.get_low());
  float16x8_t r1 = vfmsq_f16(c.get_high(), a.get_high(), b.get_high());
  return Vectorized<Half>(r0, r1);
}

#endif /* defined(aarch64) */

} // namespace CPU_CAPABILITY
} // namespace at::vec
