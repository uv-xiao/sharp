//===- HybridBridge.cpp - TL-to-RTL Bridge Implementation ----------------===//
//
// This file implements the hybrid TL-to-RTL bridge for mixed-level simulation.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/Hybrid/HybridBridge.h"
#include "sharp/Simulation/SimModule.h"
#include "sharp/Simulation/Event.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <chrono>
#include <thread>
#include <fstream>

namespace sharp {
namespace simulation {

//===----------------------------------------------------------------------===//
// HybridBridge Implementation
//===----------------------------------------------------------------------===//

HybridBridge::HybridBridge(SyncMode mode) 
    : syncMode_(mode), running_(false), tlTime_(0), rtlTime_(0),
      maxTimeDivergence_(1000) {
  stats_ = {0, 0, 0, 0, 0};
}

HybridBridge::~HybridBridge() {
  stop();
}

void HybridBridge::configure(const std::string& configFile) {
  // Load configuration from JSON file
  auto bufferOrErr = llvm::MemoryBuffer::getFile(configFile);
  if (!bufferOrErr) {
    llvm::errs() << "Failed to load bridge config: " << configFile << "\n";
    return;
  }
  
  auto json = llvm::json::parse(bufferOrErr.get()->getBuffer());
  if (!json) {
    llvm::errs() << "Failed to parse bridge config JSON\n";
    return;
  }
  
  auto& obj = *json->getAsObject();
  
  // Parse sync mode
  if (auto mode = obj.getString("sync_mode")) {
    if (*mode == "lockstep") syncMode_ = SyncMode::Lockstep;
    else if (*mode == "decoupled") syncMode_ = SyncMode::Decoupled;
    else if (*mode == "adaptive") syncMode_ = SyncMode::Adaptive;
  }
  
  // Parse max time divergence
  if (auto div = obj.getInteger("max_time_divergence")) {
    maxTimeDivergence_ = *div;
  }
  
  // Parse module mappings
  if (auto mappings = obj.getArray("module_mappings")) {
    for (auto& mapping : *mappings) {
      auto& m = *mapping.getAsObject();
      auto tl = m.getString("tl_module");
      auto rtl = m.getString("rtl_module");
      if (tl && rtl) {
        tlToRtlModuleMap_[tl->str()] = rtl->str();
      }
    }
  }
  
  // Parse method mappings
  if (auto methods = obj.getArray("method_mappings")) {
    for (auto& method : *methods) {
      auto& m = *method.getAsObject();
      auto name = m.getString("method_name");
      if (!name) continue;
      
      MethodMapping mapping;
      
      if (auto inputs = m.getArray("input_signals")) {
        for (auto& sig : *inputs) {
          if (auto s = sig.getAsString())
            mapping.inputSignals.push_back(s->str());
        }
      }
      
      if (auto outputs = m.getArray("output_signals")) {
        for (auto& sig : *outputs) {
          if (auto s = sig.getAsString())
            mapping.outputSignals.push_back(s->str());
        }
      }
      
      if (auto enable = m.getString("enable_signal"))
        mapping.enableSignal = *enable;
      
      if (auto ready = m.getString("ready_signal"))
        mapping.readySignal = *ready;
      
      methodMappings_[name->str()] = mapping;
    }
  }
}

void HybridBridge::connectModule(const std::string& tlModuleName,
                                const std::string& rtlModuleName,
                                std::shared_ptr<SimModule> tlModule) {
  tlToRtlModuleMap_[tlModuleName] = rtlModuleName;
  tlModules_[tlModuleName] = tlModule;
}

void HybridBridge::setRTLSimulator(std::unique_ptr<RTLSimulatorInterface> rtlSim) {
  rtlSimulator_ = std::move(rtlSim);
}

void HybridBridge::handleTLMethodCall(
    const std::string& moduleName,
    const std::string& methodName,
    const std::vector<uint64_t>& args,
    std::function<void(const std::vector<uint64_t>&)> callback) {
  
  BridgeEvent event;
  event.type = BridgeEventType::MethodCall;
  event.timestamp = tlTime_.load();
  event.moduleName = moduleName;
  event.methodName = methodName;
  event.arguments = args;
  event.isBlocking = true;
  
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    tlToRtlQueue_.push(event);
    stats_.methodCalls++;
  }
  
  eventCV_.notify_one();
  
  // For lockstep mode, wait for response
  if (syncMode_ == SyncMode::Lockstep) {
    // In a real implementation, we would wait for the RTL response
    // For now, simulate a response after processing
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    std::vector<uint64_t> results = {0}; // Placeholder
    callback(results);
  }
}

void HybridBridge::handleRTLEvent(const BridgeEvent& event) {
  std::lock_guard<std::mutex> lock(queueMutex_);
  rtlToTlQueue_.push(event);
  eventCV_.notify_one();
}

void HybridBridge::synchronizeTime(uint64_t tlTime) {
  tlTime_ = tlTime;
  
  if (syncMode_ == SyncMode::Lockstep) {
    // In lockstep mode, advance RTL to match TL
    while (rtlTime_ < tlTime && rtlSimulator_ && rtlSimulator_->isReady()) {
      rtlSimulator_->stepClock();
      rtlTime_++;
    }
  } else if (syncMode_ == SyncMode::Decoupled) {
    // Check if divergence exceeds threshold
    uint64_t divergence = calculateDivergence();
    if (divergence > maxTimeDivergence_) {
      // Force synchronization
      synchronizeState();
      stats_.timeSyncs++;
    }
  }
}

bool HybridBridge::needsSynchronization() const {
  if (syncMode_ == SyncMode::Lockstep) {
    return tlTime_ != rtlTime_;
  } else if (syncMode_ == SyncMode::Decoupled) {
    return calculateDivergence() > maxTimeDivergence_;
  }
  return false;
}

HybridBridge::SyncStats HybridBridge::getStatistics() const {
  std::lock_guard<std::mutex> lock(statsMutex_);
  return stats_;
}

void HybridBridge::start() {
  if (running_) return;
  
  running_ = true;
  bridgeThread_ = std::thread(&HybridBridge::bridgeWorker, this);
}

void HybridBridge::stop() {
  if (!running_) return;
  
  running_ = false;
  eventCV_.notify_all();
  
  if (bridgeThread_.joinable()) {
    bridgeThread_.join();
  }
}

bool HybridBridge::isRunning() const {
  return running_;
}

void HybridBridge::processEvents() {
  std::unique_lock<std::mutex> lock(queueMutex_);
  
  // Process TL to RTL events
  while (!tlToRtlQueue_.empty()) {
    auto event = tlToRtlQueue_.front();
    tlToRtlQueue_.pop();
    lock.unlock();
    
    switch (event.type) {
      case BridgeEventType::MethodCall: {
        // Map TL method call to RTL signals
        auto it = methodMappings_.find(event.methodName);
        if (it != methodMappings_.end() && rtlSimulator_) {
          auto& mapping = it->second;
          
          // Set input signals
          for (size_t i = 0; i < mapping.inputSignals.size() && i < event.arguments.size(); i++) {
            rtlSimulator_->setInput(mapping.inputSignals[i], event.arguments[i]);
          }
          
          // Assert enable signal
          if (!mapping.enableSignal.empty()) {
            rtlSimulator_->setInput(mapping.enableSignal, 1);
          }
          
          // Step RTL simulation
          rtlSimulator_->stepClock();
          
          // Collect outputs
          std::vector<uint64_t> results;
          for (const auto& outSig : mapping.outputSignals) {
            results.push_back(rtlSimulator_->getOutput(outSig));
          }
          
          // Create return event
          BridgeEvent retEvent;
          retEvent.type = BridgeEventType::MethodReturn;
          retEvent.timestamp = rtlTime_;
          retEvent.moduleName = event.moduleName;
          retEvent.methodName = event.methodName;
          retEvent.results = results;
          
          handleRTLEvent(retEvent);
        }
        break;
      }
      
      case BridgeEventType::StateUpdate:
        stats_.stateUpdates++;
        break;
        
      case BridgeEventType::TimeAdvance:
        stats_.timeSyncs++;
        break;
        
      default:
        break;
    }
    
    lock.lock();
  }
  
  // Process RTL to TL events
  while (!rtlToTlQueue_.empty()) {
    auto event = rtlToTlQueue_.front();
    rtlToTlQueue_.pop();
    lock.unlock();
    
    // Handle RTL events (method returns, state updates, etc.)
    
    lock.lock();
  }
}

void HybridBridge::bridgeWorker() {
  while (running_) {
    processEvents();
    
    std::unique_lock<std::mutex> lock(queueMutex_);
    if (tlToRtlQueue_.empty() && rtlToTlQueue_.empty()) {
      eventCV_.wait_for(lock, std::chrono::milliseconds(1));
    }
  }
}

void HybridBridge::mapMethodCall(const std::string& tlMethod, 
                                const std::string& rtlSignals) {
  // Parse RTL signal mapping
  // Format: "enable=en,ready=rdy,inputs=a:b:c,outputs=result"
  // This would be parsed and stored in methodMappings_
}

void HybridBridge::synchronizeState() {
  // Synchronize state between TL and RTL domains
  // This involves reading state from both sides and reconciling differences
  
  if (syncMode_ == SyncMode::Adaptive) {
    // Adjust synchronization strategy based on activity
    uint64_t divergence = calculateDivergence();
    if (divergence > maxTimeDivergence_ * 2) {
      // Switch to lockstep temporarily
      syncMode_ = SyncMode::Lockstep;
    }
  }
}

uint64_t HybridBridge::calculateDivergence() const {
  uint64_t tl = tlTime_.load();
  uint64_t rtl = rtlTime_.load();
  return (tl > rtl) ? (tl - rtl) : (rtl - tl);
}

//===----------------------------------------------------------------------===//
// ArcilatorSimulator Implementation
//===----------------------------------------------------------------------===//

struct ArcilatorSimulator::Impl {
  bool initialized = false;
  uint64_t currentTime = 0;
  std::unordered_map<std::string, uint64_t> signals;
  
  // In a real implementation, this would interface with CIRCT's arcilator
  // through its C API or by launching it as a subprocess
};

ArcilatorSimulator::ArcilatorSimulator() : pImpl(std::make_unique<Impl>()) {}

ArcilatorSimulator::~ArcilatorSimulator() = default;

bool ArcilatorSimulator::initialize(const std::string& rtlModule) {
  // In a real implementation:
  // 1. Load the Arc dialect module
  // 2. Initialize arcilator JIT engine
  // 3. Set up signal mappings
  
  pImpl->initialized = true;
  llvm::outs() << "Initialized ArcilatorSimulator with module: " << rtlModule << "\n";
  return true;
}

void ArcilatorSimulator::stepClock() {
  if (!pImpl->initialized) return;
  
  // Advance simulation by one clock cycle
  pImpl->currentTime++;
  
  // In real implementation: call arcilator step function
}

void ArcilatorSimulator::setInput(const std::string& signal, uint64_t value) {
  pImpl->signals[signal] = value;
}

uint64_t ArcilatorSimulator::getOutput(const std::string& signal) {
  auto it = pImpl->signals.find(signal);
  return (it != pImpl->signals.end()) ? it->second : 0;
}

bool ArcilatorSimulator::isReady() const {
  return pImpl->initialized;
}

uint64_t ArcilatorSimulator::getCurrentTime() const {
  return pImpl->currentTime;
}

void ArcilatorSimulator::reset() {
  pImpl->currentTime = 0;
  pImpl->signals.clear();
}

//===----------------------------------------------------------------------===//
// HybridBridgeFactory Implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<HybridBridge> HybridBridgeFactory::create(
    const std::string& config,
    SyncMode mode) {
  auto bridge = std::make_unique<HybridBridge>(mode);
  if (!config.empty()) {
    bridge->configure(config);
  }
  return bridge;
}

std::unique_ptr<RTLSimulatorInterface> HybridBridgeFactory::createRTLSimulator(
    const std::string& type) {
  if (type == "arcilator") {
    return std::make_unique<ArcilatorSimulator>();
  }
  // Add other RTL simulator types here
  return nullptr;
}

} // namespace simulation
} // namespace sharp