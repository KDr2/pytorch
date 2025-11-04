#include <gtest/gtest.h>

#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

namespace {
template <typename T>
inline void expectThrowsEq(T&& fn, const char* expected_msg) {
  try {
    std::forward<T>(fn)();
  } catch (const c10::Error& e) {
    EXPECT_STREQ(e.what_without_backtrace(), expected_msg);
    return;
  }
  ADD_FAILURE() << "Expected to throw exception with message \"" << expected_msg
                << "\" but didn't throw";
}
} // namespace

TEST(NofatalTest, TorchCheckComparisons) {
  expectThrowsEq(
      []() { TORCH_CHECK_EQ(1, 2) << "i am a silly message " << 1; },
      "Check failed: 1 == 2 (1 vs. 2). i am a silly message 1");
  expectThrowsEq(
      []() { TORCH_CHECK_NE(2, 2); }, "Check failed: 2 != 2 (2 vs. 2). ");
  expectThrowsEq(
      []() { TORCH_CHECK_LT(2, 2); }, "Check failed: 2 < 2 (2 vs. 2). ");
  expectThrowsEq(
      []() { TORCH_CHECK_LE(3, 2); }, "Check failed: 3 <= 2 (3 vs. 2). ");
  expectThrowsEq(
      []() { TORCH_CHECK_GT(2, 2); }, "Check failed: 2 > 2 (2 vs. 2). ");
  expectThrowsEq(
      []() { TORCH_CHECK_GE(2, 3); }, "Check failed: 2 >= 3 (2 vs. 3). ");
  expectThrowsEq(
      []() {
        void* p = nullptr;
        TORCH_CHECK_NOTNULL(p);
      },
      "Check failed: 'p' must be non NULL. ");
}

// Death tests are not supported on iOS and some other platforms
#if GTEST_HAS_DEATH_TEST
TEST(NofatalTest, RegularCheckAborts) {
  // CHECK is available in both glog and non-glog configurations
  EXPECT_DEATH({ CHECK(false) << "This should abort"; }, "Check failed");

#ifdef C10_USE_GLOG
  // CHECK_EQ and CHECK_NOTNULL are only available when using glog
  EXPECT_DEATH({ CHECK_EQ(1, 2) << "This should abort"; }, "Check failed");
  EXPECT_DEATH(
      {
        void* p = nullptr;
        CHECK_NOTNULL(p);
      },
      "");
#endif
}
#endif // GTEST_HAS_DEATH_TEST
