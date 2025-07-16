//===- ScheduleDebugger.h - Schedule and Conflict Matrix Debug Output ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ScheduleDebugger class for pretty-printing schedule
// and conflict matrix information during TxnToFIRRTL conversion.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_ANALYSIS_SCHEDULEDEBUGGER_H
#define SHARP_ANALYSIS_SCHEDULEDEBUGGER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace mlir {
namespace sharp {

/// Structure to represent schedule information for debug output
struct ScheduleDebugInfo {
  std::vector<std::string> actions;
  std::string moduleName;
  std::string timingMode;
};

/// Structure to represent conflict matrix information for debug output
struct ConflictMatrixDebugInfo {
  llvm::StringMap<std::string> conflictEntries; // "action1,action2" -> "SB/SA/C/CF"
  std::vector<std::string> allActions;
  std::string moduleName;
};

/// Debugger class for pretty-printing schedule and conflict matrix information
class ScheduleDebugger {
public:
  ScheduleDebugger(llvm::StringRef passName, llvm::StringRef timingMode);
  
  /// Set schedule information
  void setSchedule(const ScheduleDebugInfo& schedule);
  
  /// Set conflict matrix information
  void setConflictMatrix(const ConflictMatrixDebugInfo& conflictMatrix);
  
  /// Pretty-print the schedule as a table
  void printSchedule(llvm::raw_ostream& os = llvm::errs()) const;
  
  /// Pretty-print the conflict matrix as a 2D table
  void printConflictMatrix(llvm::raw_ostream& os = llvm::errs()) const;
  
  /// Print both schedule and conflict matrix with headers
  void printAll(llvm::raw_ostream& os = llvm::errs()) const;
  
  /// Print a summary header for the module
  void printModuleHeader(llvm::StringRef moduleName, llvm::raw_ostream& os = llvm::errs()) const;

private:
  std::string passName;
  std::string timingMode;
  ScheduleDebugInfo scheduleInfo;
  ConflictMatrixDebugInfo conflictMatrixInfo;
  
  /// Helper function to get conflict relationship string
  std::string getConflictString(llvm::StringRef action1, llvm::StringRef action2) const;
  
  /// Helper function to format table borders
  void printTableBorder(llvm::raw_ostream& os, const std::vector<size_t>& columnWidths) const;
  
  /// Helper function to format table row
  void printTableRow(llvm::raw_ostream& os, const std::vector<std::string>& cells, 
                     const std::vector<size_t>& columnWidths) const;
  
  /// Helper function to calculate maximum column widths
  std::vector<size_t> calculateColumnWidths(const std::vector<std::vector<std::string>>& table) const;
};

} // namespace sharp
} // namespace mlir

#endif // SHARP_ANALYSIS_SCHEDULEDEBUGGER_H