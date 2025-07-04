//===- Channel.h - DAM-style Time-Bridging Channel ------------------------===//
//
// This file defines the Channel class for communication between contexts
// in DAM-based concurrent simulation. Channels handle time-bridging and
// synchronization between contexts at different simulated times.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_CONCURRENT_CHANNEL_H
#define SHARP_SIMULATION_CONCURRENT_CHANNEL_H

#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace sharp {
namespace sim {
namespace concurrent {

/// Timestamped data for channel communication
template <typename T>
struct TimestampedData {
  uint64_t timestamp;
  T data;
  
  TimestampedData(uint64_t ts, T d) : timestamp(ts), data(std::move(d)) {}
};

/// Time-bridging channel following DAM methodology
template <typename T>
class Channel {
public:
  /// Create a channel with optional bounded capacity (0 = unbounded)
  explicit Channel(size_t capacity = 0) 
    : capacity(capacity), senderTime(0), receiverTime(0) {}

  /// Enqueue data with timestamp (may block if bounded and full)
  void enqueue(const T& data, uint64_t timestamp) {
    std::unique_lock<std::mutex> lock(mutex);
    
    // Wait if bounded channel is full
    if (capacity > 0) {
      senderCv.wait(lock, [this] { return buffer.size() < capacity; });
    }
    
    buffer.emplace_back(timestamp, data);
    senderTime.store(timestamp, std::memory_order_release);
    
    // Notify receiver
    receiverCv.notify_one();
  }

  /// Dequeue data (may block if empty)
  std::optional<T> dequeue(uint64_t currentTime) {
    std::unique_lock<std::mutex> lock(mutex);
    
    // Wait for data at or before current time
    receiverCv.wait(lock, [this, currentTime] {
      return !buffer.empty() && buffer.front().timestamp <= currentTime;
    });
    
    if (!buffer.empty() && buffer.front().timestamp <= currentTime) {
      T data = std::move(buffer.front().data);
      uint64_t ts = buffer.front().timestamp;
      buffer.pop_front();
      
      receiverTime.store(ts, std::memory_order_release);
      
      // Notify sender if bounded
      if (capacity > 0) {
        senderCv.notify_one();
      }
      
      return data;
    }
    
    return std::nullopt;
  }

  /// Peek at next data without removing (non-blocking)
  std::optional<std::pair<uint64_t, T>> peekNext() const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!buffer.empty()) {
      const auto& front = buffer.front();
      return std::make_pair(front.timestamp, front.data);
    }
    return std::nullopt;
  }

  /// Get sender's current time (for synchronization)
  uint64_t getSenderTime() const {
    return senderTime.load(std::memory_order_acquire);
  }

  /// Get receiver's current time (for synchronization)
  uint64_t getReceiverTime() const {
    return receiverTime.load(std::memory_order_acquire);
  }

  /// Check if channel is empty
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex);
    return buffer.empty();
  }

  /// Get current size
  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return buffer.size();
  }

private:
  mutable std::mutex mutex;
  std::condition_variable senderCv;    // For blocking senders
  std::condition_variable receiverCv;  // For blocking receivers
  
  std::deque<TimestampedData<T>> buffer;
  size_t capacity; // 0 = unbounded
  
  // Atomic timestamps for lock-free synchronization checks
  std::atomic<uint64_t> senderTime;
  std::atomic<uint64_t> receiverTime;
};

} // namespace concurrent
} // namespace sim
} // namespace sharp

#endif // SHARP_SIMULATION_CONCURRENT_CHANNEL_H