#include "sharp/Analysis/ConflictDebugger.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

namespace mlir {
namespace sharp {

ConflictDebugger::ConflictDebugger(llvm::StringRef passName, TimingMode mode)
    : passName(passName), mode(mode) {
  currentWillFire.mode = mode;
}

ConflictDebugger& ConflictDebugger::logWillFireDecision(
    llvm::StringRef actionName,
    bool enabled,
    llvm::StringRef abortCondition,
    const std::vector<ConflictDebugInfo>& conflicts,
    bool finalResult) {
  
  currentWillFire.actionName = actionName.str();
  currentWillFire.enabled = enabled;
  currentWillFire.reachAbort = !abortCondition.empty() && abortCondition != "0";
  currentWillFire.abortCondition = abortCondition.str();
  currentWillFire.conflicts = conflicts;
  currentWillFire.finalWillFire = finalResult;
  
  // Conflict status is now determined per ConflictDebugInfo object
  // Each conflict provides its own explanation of the conflict type
  currentWillFire.conflictsWithEarlier = !conflicts.empty();
  currentWillFire.conflictInside = false; // Deprecated - conflict type in each ConflictDebugInfo
  
  currentType = DebugInfoType::WillFireGeneration;
  return *this;
}

ConflictDebugger& ConflictDebugger::logConflictDetection(
    llvm::StringRef method1,
    llvm::StringRef method2,
    llvm::StringRef conflictType,
    llvm::StringRef reachCondition1,
    llvm::StringRef reachCondition2,
    bool isConflicting) {
  
  ConflictDebugInfo conflict;
  conflict.method1 = method1.str();
  conflict.method2 = method2.str();
  conflict.conflictType = conflictType.str();
  conflict.reachCondition1 = reachCondition1.str();
  conflict.reachCondition2 = reachCondition2.str();
  conflict.isConflicting = isConflicting;
  
  if (isConflicting) {
    conflict.explanation = "Methods " + method1.str() + " and " + method2.str() + 
                          " have " + conflictType.str() + " conflict relationship";
    if (!reachCondition1.empty() && reachCondition1 != "1") {
      conflict.explanation += ", " + method1.str() + " reachable when: " + reachCondition1.str();
    }
    if (!reachCondition2.empty() && reachCondition2 != "1") {
      conflict.explanation += ", " + method2.str() + " reachable when: " + reachCondition2.str();
    }
  } else {
    conflict.explanation = "Methods " + method1.str() + " and " + method2.str() + 
                          " are conflict-free (" + conflictType.str() + ")";
  }
  
  currentConflicts.push_back(conflict);
  currentType = DebugInfoType::ConflictDetection;
  return *this;
}

ConflictDebugger& ConflictDebugger::logReachabilityResult(
    llvm::StringRef methodCall,
    llvm::StringRef reachCondition,
    bool isReachable) {
  
  currentExplanation = "Method call " + methodCall.str() + 
                      (isReachable ? " is reachable" : " is not reachable");
  if (!reachCondition.empty() && reachCondition != "1") {
    currentExplanation += " under condition: " + reachCondition.str();
  }
  
  currentType = DebugInfoType::ReachabilityAnalysis;
  return *this;
}

ConflictDebugger& ConflictDebugger::logAbortCondition(
    llvm::StringRef actionName,
    llvm::StringRef abortCondition,
    llvm::StringRef derivation) {
  
  currentExplanation = "[" + passName.str() + "] Abort analysis for action " + 
                      actionName.str() + ": " + abortCondition.str();
  if (!derivation.empty()) {
    currentExplanation += "\n  Derivation: " + derivation.str();
  }
  
  currentType = DebugInfoType::AbortCondition;
  return *this;
}

ConflictDebugger& ConflictDebugger::logMethodTracking(
    llvm::StringRef methodName,
    bool wasCalled,
    llvm::StringRef callingAction) {
  
  currentExplanation = "Method " + methodName.str() + 
                      (wasCalled ? " was called" : " was not called");
  if (!callingAction.empty()) {
    currentExplanation += " by action " + callingAction.str();
  }
  
  currentType = DebugInfoType::MethodTracking;
  return *this;
}

ConflictDebugger& ConflictDebugger::setExplanation(llvm::StringRef explanation) {
  currentExplanation = explanation.str();
  return *this;
}

ConflictDebugger& ConflictDebugger::setType(DebugInfoType type) {
  currentType = type;
  return *this;
}

void ConflictDebugger::emit(llvm::raw_ostream& os) {
  os << "[" << passName << " Debug] " << getTypeString() << " (" << getModeString() << " mode)\n";
  
  switch (currentType) {
    case DebugInfoType::WillFireGeneration:
      formatWillFireInfo(os, currentWillFire);
      break;
    
    case DebugInfoType::ConflictDetection:
      for (const auto& conflict : currentConflicts) {
        formatConflictInfo(os, conflict);
      }
      break;
    
    default:
      os << "  " << currentExplanation << "\n";
      break;
  }
  
  os << "\n";
  
  // Clear current state
  currentConflicts.clear();
  currentExplanation.clear();
  currentWillFire = WillFireDebugInfo{};
  currentWillFire.mode = mode;
}

void ConflictDebugger::emitToStderr() {
  emit(llvm::errs());
}

std::string ConflictDebugger::createWillFireSummary(const WillFireDebugInfo& info) {
  std::string summary;
  llvm::raw_string_ostream os(summary);
  
  os << "=== Will-Fire Analysis Summary ===\n";
  os << "Action: " << info.actionName << " (" << getModeString() << " mode)\n";
  os << "Enabled: " << (info.enabled ? "YES" : "NO") << "\n";
  os << "Abort Condition: " << (info.reachAbort ? "PRESENT" : "NONE");
  if (info.reachAbort && !info.abortCondition.empty()) {
    os << " (" << info.abortCondition << ")";
  }
  os << "\n";
  os << "Conflicts with Earlier: " << (info.conflictsWithEarlier ? "YES" : "NO") << "\n";
  if (mode == TimingMode::Static) {
    os << "Conflicts Inside Action: " << (info.conflictInside ? "YES" : "NO") << "\n";
  }
  os << "Final Will-Fire: " << (info.finalWillFire ? "YES" : "NO") << "\n";
  
  if (!info.conflicts.empty()) {
    os << "\nConflict Details:\n";
    for (const auto& conflict : info.conflicts) {
      os << "  " << conflict.method1 << " vs " << conflict.method2 
         << " = " << conflict.conflictType 
         << (conflict.isConflicting ? " (CONFLICT)" : " (OK)") << "\n";
    }
  }
  
  return os.str();
}

std::string ConflictDebugger::createConflictSummary(const std::vector<ConflictDebugInfo>& conflicts) {
  std::string summary;
  llvm::raw_string_ostream os(summary);
  
  os << "=== Conflict Analysis Summary ===\n";
  os << "Total conflicts checked: " << conflicts.size() << "\n";
  
  int conflictCount = 0;
  for (const auto& conflict : conflicts) {
    if (conflict.isConflicting) conflictCount++;
  }
  
  os << "Actual conflicts found: " << conflictCount << "\n";
  
  if (conflictCount > 0) {
    os << "\nConflicting pairs:\n";
    for (const auto& conflict : conflicts) {
      if (conflict.isConflicting) {
        os << "  " << conflict.method1 << " " << conflict.conflictType 
           << " " << conflict.method2 << "\n";
      }
    }
  }
  
  return os.str();
}

llvm::StringRef ConflictDebugger::getModeString() const {
  switch (mode) {
    case TimingMode::Static: return "static";
    case TimingMode::Dynamic: return "dynamic";
  }
  return "unknown";
}

llvm::StringRef ConflictDebugger::getTypeString() const {
  switch (currentType) {
    case DebugInfoType::WillFireGeneration: return "Will-Fire Generation";
    case DebugInfoType::ConflictDetection: return "Conflict Detection";
    case DebugInfoType::ReachabilityAnalysis: return "Reachability Analysis";
    case DebugInfoType::AbortCondition: return "Abort Condition";
    case DebugInfoType::MethodTracking: return "Method Tracking";
    case DebugInfoType::ScheduleEvaluation: return "Schedule Evaluation";
  }
  return "Unknown";
}

void ConflictDebugger::formatConflictInfo(llvm::raw_ostream& os, const ConflictDebugInfo& conflict) {
  os << "  Conflict Check: " << conflict.method1 << " vs " << conflict.method2 << "\n";
  os << "    Relationship: " << conflict.conflictType << "\n";
  os << "    Result: " << (conflict.isConflicting ? "CONFLICT" : "OK") << "\n";
  if (!conflict.reachCondition1.empty() && conflict.reachCondition1 != "1") {
    os << "    " << conflict.method1 << " reach condition: " << conflict.reachCondition1 << "\n";
  }
  if (!conflict.reachCondition2.empty() && conflict.reachCondition2 != "1") {
    os << "    " << conflict.method2 << " reach condition: " << conflict.reachCondition2 << "\n";
  }
  os << "    Explanation: " << conflict.explanation << "\n";
}

void ConflictDebugger::formatWillFireInfo(llvm::raw_ostream& os, const WillFireDebugInfo& info) {
  os << "  Action: " << info.actionName << "\n";
  os << "    Enabled: " << (info.enabled ? "YES" : "NO") << "\n";
  os << "    Abort condition: " << (info.reachAbort ? "PRESENT" : "NONE");
  if (!info.abortCondition.empty()) {
    os << " (" << info.abortCondition << ")";
  }
  os << "\n";
  
  if (mode == TimingMode::Static) {
    os << "    Conflicts with earlier: " << (info.conflictsWithEarlier ? "YES" : "NO") << "\n";
    os << "    Conflicts inside action: " << (info.conflictInside ? "YES" : "NO") << "\n";
  } else {
    os << "    Method conflicts: " << (info.conflictsWithEarlier ? "YES" : "NO") << "\n";
  }
  
  os << "    Final will-fire: " << (info.finalWillFire ? "YES" : "NO") << "\n";
  
  if (!info.conflicts.empty()) {
    os << "    Conflict details:\n";
    for (const auto& conflict : info.conflicts) {
      os << "      " << conflict.method1 << " " << conflict.conflictType 
         << " " << conflict.method2 << " = " << (conflict.isConflicting ? "CONFLICT" : "OK") << "\n";
    }
  }
  
  if (!info.explanation.empty()) {
    os << "    Explanation: " << info.explanation << "\n";
  }
}

} // namespace sharp
} // namespace mlir