#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "lc_keyframe_db.hpp"

/*
 * LCWorker — bounded-queue background thread that consumes LCKeyframe
 * snapshots produced by the main loop and hands them off to a user-supplied
 * processing callback (detection + GICP + PGO + correction, installed in
 * later steps).
 *
 * Backpressure policy: drop-oldest. If the queue fills up, the next
 * enqueue evicts the front element and a dropped-counter is bumped. LC is
 * best-effort — missing a keyframe is always better than stalling the main
 * odometry loop.
 *
 * Lifecycle: `start()` from the node constructor, `stop()` from the
 * destructor. Worker exits cleanly on stop() after draining an in-flight
 * callback invocation (no long-lived state across dequeue calls).
 *
 * Concurrency:
 *   - `enqueue()` is called from the main thread (map_incremental).
 *   - `set_callback()` is called before `start()` and must not be changed
 *     while the worker is running.
 *   - The callback runs on the worker thread.
 */
class LCWorker {
public:
    using Callback = std::function<void(const LCKeyframe&)>;

    LCWorker() = default;

    LCWorker(const LCWorker&) = delete;
    LCWorker& operator=(const LCWorker&) = delete;

    ~LCWorker() { stop(); }

    void set_callback(Callback cb) { callback_ = std::move(cb); }
    void set_max_queue_size(size_t n) { max_queue_size_ = n; }

    void start()
    {
        if (running_.exchange(true)) return;  // idempotent
        thread_ = std::thread([this] { this->loop(); });
    }

    void stop()
    {
        if (!running_.exchange(false)) return;  // idempotent
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void enqueue(LCKeyframe kf)
    {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (queue_.size() >= max_queue_size_) {
                queue_.pop_front();
                ++dropped_count_;
            }
            queue_.push_back(std::move(kf));
        }
        cv_.notify_one();
    }

    // Introspection (metrics)
    size_t queue_depth() const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }

    size_t dropped_count() const { return dropped_count_.load(); }
    size_t processed_count() const { return processed_count_.load(); }

private:
    void loop()
    {
        while (running_.load()) {
            LCKeyframe kf;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait_for(lock, std::chrono::milliseconds(200), [this] {
                    return !queue_.empty() || !running_.load();
                });
                if (!running_.load()) return;
                if (queue_.empty()) continue;
                kf = std::move(queue_.front());
                queue_.pop_front();
            }
            if (callback_) callback_(kf);
            ++processed_count_;
        }
    }

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<LCKeyframe> queue_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<size_t> processed_count_{0};
    std::atomic<size_t> dropped_count_{0};
    size_t max_queue_size_ = 32;
    Callback callback_;
};
