#include <exception>
#include <future>
#include <stdexcept>

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/Healthcheck.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d {

Healthcheck::Healthcheck(
    bool abortOnError,
    std::chrono::milliseconds interval,
    std::chrono::milliseconds timeout)
    : abortOnError_(abortOnError), interval_(interval), timeout_(timeout) {
  worker_ = std::async(std::launch::async, [this]() {
    try {
      runLoop();
    } catch (const std::exception& e) {
      C10D_ERROR("Healthcheck thread failed: {}", e.what());
    } catch (...) {
      C10D_ERROR("Healthcheck thread failed with unknown exception");
    }
  });
}

void Healthcheck::runLoop() {
  C10D_INFO("Healthcheck setup...");
  for (int side = 0; side < 2; side++) {
    setup(side);
  }

  while (true) {
    C10D_INFO("Running healthchecks...");

    std::vector<std::future<void>> futures;
    futures.reserve(2);

    for (int side = 0; side < 2; side++) {
      futures.emplace_back(std::async(
          std::launch::async,
          [](Healthcheck* self, int side) { self->runHealthcheck(side); },
          this,
          side));
    }

    // calculate deadline for the futures
    std::chrono::time_point<std::chrono::system_clock> deadline =
        std::chrono::system_clock::now() + timeout_;

    int failures = 0;

    // wait for futures to complete
    for (auto& future : futures) {
      auto status = future.wait_until(deadline);
      if (status == std::future_status::timeout) {
        failures += 1;
        C10D_ERROR("Healthcheck timed out");
        continue;
      }
      TORCH_INTERNAL_ASSERT(status == std::future_status::ready);

      try {
        future.get();
        C10D_INFO("Healthcheck passed");
      } catch (const std::exception& e) {
        C10D_WARNING("Healthcheck failed: {}", e.what());
        failures += 1;
        continue;
      } catch (...) {
        C10D_WARNING("Healthcheck failed with unknown exception");
        failures += 1;
        continue;
      }
    }

    if (failures > 0) {
      C10D_WARNING("Healthchecks had {} failures", failures);
    } else {
      C10D_INFO("All healthchecks passed");
    }
    numFailures_ = failures;
    if (failures == 2) {
      C10D_ERROR("Current host identified as problematic!");
      if (abortOnError_) {
        std::abort();
      }
      Healthcheck::shutdown();
      return;
    }

    // wait for interval
    waitFor(interval_);
    if (isShutdown()) {
      throw std::runtime_error("shutting down");
    }
  }
}

void Healthcheck::waitFor(std::chrono::milliseconds duration) {
  std::unique_lock lock{shutdownM_};
  if (shutdown_) {
    return;
  }
  shutdownCv_.wait_for(lock, duration);
}

bool Healthcheck::isShutdown() {
  std::unique_lock lock{shutdownM_};
  return shutdown_;
}

void Healthcheck::shutdown() {
  C10D_INFO("shutting down...");
  {
    std::unique_lock lock{shutdownM_};
    shutdown_ = true;
  }
  shutdownCv_.notify_all();
}

void Healthcheck::wait() {
  worker_.wait();
}

} // namespace c10d
