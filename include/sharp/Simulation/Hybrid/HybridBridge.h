//===- HybridBridge.h - TL-to-RTL Bridge Interface -----------------------===//
//
// This file defines the bridge interface for hybrid TL-to-RTL simulation.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_HYBRID_HYBRIDBRIDGE_H
#define SHARP_SIMULATION_HYBRID_HYBRIDBRIDGE_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sharp {
namespace simulation {

/// Forward declarations
class SimModule;
class Event;

/// Bridge event types for synchronization
enum class BridgeEventType {
  MethodCall,      // TL method call to RTL
  MethodReturn,    // RTL method return to TL
  StateUpdate,     // State synchronization
  TimeAdvance,     // Time synchronization
  Reset,           // Reset signal
  Clock            // Clock edge
};

/// Bridge event structure
struct BridgeEvent {
  BridgeEventType type;
  uint64_t timestamp;
  std::string moduleName;
  std::string methodName;
  std::vector<uint64_t> arguments;
  std::vector<uint64_t> results;
  bool isBlocking;
};

/// Synchronization mode between TL and RTL
enum class SyncMode {
  Lockstep,        // TL and RTL advance together
  Decoupled,       // TL and RTL can diverge within bounds
  Adaptive         // Dynamically adjust based on activity
};

/// Interface for RTL simulation backend (e.g., arcilator)
class RTLSimulatorInterface {
public:
  virtual ~RTLSimulatorInterface() = default;
  
  /// Initialize the RTL simulator with the compiled design
  virtual bool initialize(const std::string& rtlModule) = 0;
  
  /// Advance RTL simulation by one clock cycle
  virtual void stepClock() = 0;
  
  /// Set input signal value
  virtual void setInput(const std::string& signal, uint64_t value) = 0;
  
  /// Get output signal value
  virtual uint64_t getOutput(const std::string& signal) = 0;
  
  /// Check if RTL simulator is ready
  virtual bool isReady() const = 0;
  
  /// Get current RTL simulation time
  virtual uint64_t getCurrentTime() const = 0;
  
  /// Reset the RTL design
  virtual void reset() = 0;
};

/// TL-to-RTL bridge for hybrid simulation
class HybridBridge {
public:
  HybridBridge(SyncMode mode = SyncMode::Lockstep);
  ~HybridBridge();
  
  /// Configure the bridge
  void configure(const std::string& configFile);
  
  /// Connect TL module to RTL implementation
  void connectModule(const std::string& tlModuleName,
                    const std::string& rtlModuleName,
                    std::shared_ptr<SimModule> tlModule);
  
  /// Set the RTL simulator backend
  void setRTLSimulator(std::unique_ptr<RTLSimulatorInterface> rtlSim);
  
  /// Handle TL method call
  void handleTLMethodCall(const std::string& moduleName,
                         const std::string& methodName,
                         const std::vector<uint64_t>& args,
                         std::function<void(const std::vector<uint64_t>&)> callback);
  
  /// Handle RTL event
  void handleRTLEvent(const BridgeEvent& event);
  
  /// Synchronize TL and RTL time
  void synchronizeTime(uint64_t tlTime);
  
  /// Check if synchronization is needed
  bool needsSynchronization() const;
  
  /// Get synchronization statistics
  struct SyncStats {
    uint64_t methodCalls;
    uint64_t stateUpdates;
    uint64_t timeSyncs;
    uint64_t avgLatency;
    uint64_t maxDivergence;
  };
  SyncStats getStatistics() const;
  
  /// Start the bridge execution
  void start();
  
  /// Stop the bridge execution
  void stop();
  
  /// Check if bridge is running
  bool isRunning() const;

private:
  /// Process pending bridge events
  void processEvents();
  
  /// Worker thread for bridge processing
  void bridgeWorker();
  
  /// Map method calls between TL and RTL
  void mapMethodCall(const std::string& tlMethod, const std::string& rtlSignals);
  
  /// Synchronize state between domains
  void synchronizeState();
  
  /// Calculate time divergence
  uint64_t calculateDivergence() const;
  
private:
  SyncMode syncMode_;
  std::atomic<bool> running_;
  
  // Module mappings
  std::unordered_map<std::string, std::string> tlToRtlModuleMap_;
  std::unordered_map<std::string, std::shared_ptr<SimModule>> tlModules_;
  
  // Method mappings
  struct MethodMapping {
    std::vector<std::string> inputSignals;
    std::vector<std::string> outputSignals;
    std::string enableSignal;
    std::string readySignal;
  };
  std::unordered_map<std::string, MethodMapping> methodMappings_;
  
  // Event queues
  std::queue<BridgeEvent> tlToRtlQueue_;
  std::queue<BridgeEvent> rtlToTlQueue_;
  
  // Synchronization primitives
  mutable std::mutex queueMutex_;
  std::condition_variable eventCV_;
  std::thread bridgeThread_;
  
  // RTL simulator interface
  std::unique_ptr<RTLSimulatorInterface> rtlSimulator_;
  
  // Time management
  std::atomic<uint64_t> tlTime_;
  std::atomic<uint64_t> rtlTime_;
  uint64_t maxTimeDivergence_;
  
  // Statistics
  mutable std::mutex statsMutex_;
  SyncStats stats_;
};

/// Arcilator-based RTL simulator implementation
class ArcilatorSimulator : public RTLSimulatorInterface {
public:
  ArcilatorSimulator();
  ~ArcilatorSimulator() override;
  
  bool initialize(const std::string& rtlModule) override;
  void stepClock() override;
  void setInput(const std::string& signal, uint64_t value) override;
  uint64_t getOutput(const std::string& signal) override;
  bool isReady() const override;
  uint64_t getCurrentTime() const override;
  void reset() override;

private:
  struct Impl;
  std::unique_ptr<Impl> pImpl;
};

/// Factory for creating hybrid bridges
class HybridBridgeFactory {
public:
  static std::unique_ptr<HybridBridge> create(
      const std::string& config,
      SyncMode mode = SyncMode::Lockstep);
  
  static std::unique_ptr<RTLSimulatorInterface> createRTLSimulator(
      const std::string& type);
};

} // namespace simulation
} // namespace sharp

#endif // SHARP_SIMULATION_HYBRID_HYBRIDBRIDGE_H