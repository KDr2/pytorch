#pragma once

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include <c10/macros/Export.h>
#include <torch/custom_class.h>

namespace c10d {

class TORCH_API Healthcheck : public torch::CustomClassHolder {
 public:
  Healthcheck(
      bool abortOnError = false,
      std::chrono::milliseconds interval = std::chrono::seconds(10),
      std::chrono::milliseconds timeout = std::chrono::seconds(10));
  virtual ~Healthcheck() = default;

  virtual void shutdown();
  void wait();

  int getNumFailures() {
    return numFailures_;
  }

 protected:
  void waitFor(std::chrono::milliseconds duration);
  bool isShutdown();

 private:
  // Called to setup each side, this is run on the worker thread.
  virtual void setup(int side) = 0;

  // Called in an individual thread to run the healthcheck.
  virtual void runHealthcheck(int side) = 0;

  void runLoop();

 protected:
  const bool abortOnError_;
  const std::chrono::milliseconds interval_;
  const std::chrono::milliseconds timeout_;

 private:
  std::atomic<int> numFailures_{-1};
  std::future<void> worker_{};

  std::mutex shutdownM_;
  std::condition_variable shutdownCv_;
  bool shutdown_{false};
};

} // namespace c10d
