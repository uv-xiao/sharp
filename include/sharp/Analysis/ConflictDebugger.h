#ifndef SHARP_ANALYSIS_CONFLICTDEBUGGER_H
#define SHARP_ANALYSIS_CONFLICTDEBUGGER_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace sharp {

/// Enum for different debugging information types
enum class DebugInfoType {
  WillFireGeneration,
  ConflictDetection,
  ReachabilityAnalysis,
  AbortCondition,
  MethodTracking,
  ScheduleEvaluation
};

/// Enum for timing modes
enum class TimingMode {
  Static,
  Dynamic
};

/// Structure to hold conflict debug information
struct ConflictDebugInfo {
  std::string actionName;
  std::string method1;
  std::string method2;
  std::string conflictType; // "SB", "SA", "C", "CF"
  std::string reachCondition1;
  std::string reachCondition2;
  bool isConflicting;
  std::string explanation;
};

/// Structure to hold will-fire debug information
struct WillFireDebugInfo {
  std::string actionName;
  TimingMode mode;
  bool enabled;
  bool reachAbort;
  bool conflictsWithEarlier;
  bool conflictInside;
  bool finalWillFire;
  std::string enableCondition;
  std::string abortCondition;
  std::vector<ConflictDebugInfo> conflicts;
  std::string explanation;
};

/// Utility class for structured debugging output for conflict resolution
/// 
/// Usage:
///   ConflictDebugger debugger("TxnToFIRRTL", TimingMode::Dynamic);
///   debugger.logWillFireDecision(action, enabled, abortCondition, conflicts, finalResult)
///           .setExplanation("Action blocked due to conflict with earlier action")
///           .emit(llvm::outs());
class ConflictDebugger {
public:
  ConflictDebugger(llvm::StringRef passName, TimingMode mode);

  /// Log a will-fire decision for an action
  ConflictDebugger& logWillFireDecision(
    llvm::StringRef actionName,
    bool enabled,
    llvm::StringRef abortCondition,
    const std::vector<ConflictDebugInfo>& conflicts,
    bool finalResult);

  /// Log conflict detection between two methods
  ConflictDebugger& logConflictDetection(
    llvm::StringRef method1,
    llvm::StringRef method2,
    llvm::StringRef conflictType,
    llvm::StringRef reachCondition1,
    llvm::StringRef reachCondition2,
    bool isConflicting);

  /// Log reachability analysis result
  ConflictDebugger& logReachabilityResult(
    llvm::StringRef methodCall,
    llvm::StringRef reachCondition,
    bool isReachable);

  /// Log abort condition calculation
  ConflictDebugger& logAbortCondition(
    llvm::StringRef actionName,
    llvm::StringRef abortCondition,
    llvm::StringRef derivation);

  /// Log method tracking for dynamic/most-dynamic modes
  ConflictDebugger& logMethodTracking(
    llvm::StringRef methodName,
    bool wasCalled,
    llvm::StringRef callingAction);

  /// Set explanation for current debug entry
  ConflictDebugger& setExplanation(llvm::StringRef explanation);

  /// Set debug info type
  ConflictDebugger& setType(DebugInfoType type);

  /// Emit debug information to output stream
  void emit(llvm::raw_ostream& os);

  /// Emit debug information to stderr
  void emitToStderr();

  /// Create a formatted summary for a complete will-fire analysis
  std::string createWillFireSummary(const WillFireDebugInfo& info);

  /// Create a formatted conflict analysis summary
  std::string createConflictSummary(const std::vector<ConflictDebugInfo>& conflicts);

private:
  llvm::StringRef passName;
  TimingMode mode;
  DebugInfoType currentType = DebugInfoType::WillFireGeneration;
  
  // Current debug state
  WillFireDebugInfo currentWillFire;
  std::vector<ConflictDebugInfo> currentConflicts;
  std::string currentExplanation;
  
  // Helper methods
  llvm::StringRef getModeString() const;
  llvm::StringRef getTypeString() const;
  void formatConflictInfo(llvm::raw_ostream& os, const ConflictDebugInfo& conflict);
  void formatWillFireInfo(llvm::raw_ostream& os, const WillFireDebugInfo& info);
};

/// Macro for conditional debugging output
#define CONFLICT_DEBUG_ENABLED() (getenv("SHARP_DEBUG_CONFLICTS") != nullptr)

#define CONFLICT_DEBUG(debugger, expr) \
  do { \
    if (CONFLICT_DEBUG_ENABLED()) { \
      (debugger).expr.emitToStderr(); \
    } \
  } while (0)

} // namespace sharp
} // namespace mlir

#endif // SHARP_ANALYSIS_CONFLICTDEBUGGER_H