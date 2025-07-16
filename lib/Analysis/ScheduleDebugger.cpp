//===- ScheduleDebugger.cpp - Schedule and Conflict Matrix Debug Output -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDebugger class for pretty-printing schedule
// and conflict matrix information during TxnToFIRRTL conversion.
//
//===----------------------------------------------------------------------===//

#include "sharp/Analysis/ScheduleDebugger.h"
#include "llvm/Support/Format.h"
#include <algorithm>
#include <iomanip>

namespace mlir {
namespace sharp {

ScheduleDebugger::ScheduleDebugger(llvm::StringRef passName, llvm::StringRef timingMode)
    : passName(passName.str()), timingMode(timingMode.str()) {}

void ScheduleDebugger::setSchedule(const ScheduleDebugInfo& schedule) {
  scheduleInfo = schedule;
}

void ScheduleDebugger::setConflictMatrix(const ConflictMatrixDebugInfo& conflictMatrix) {
  conflictMatrixInfo = conflictMatrix;
}

void ScheduleDebugger::printSchedule(llvm::raw_ostream& os) const {
  if (scheduleInfo.actions.empty()) {
    os << "  No schedule information available\n";
    return;
  }
  
  os << "+----------------- SCHEDULE -----------------+\n";
  os << "| Index | Action Name                       |\n";
  os << "+-------+-----------------------------------+\n";
  
  for (size_t i = 0; i < scheduleInfo.actions.size(); ++i) {
    os << "| " << llvm::format("%5zu", i) << " | " 
       << llvm::format("%-33s", scheduleInfo.actions[i].c_str()) << " |\n";
  }
  
  os << "+-------+-----------------------------------+\n";
}

void ScheduleDebugger::printConflictMatrix(llvm::raw_ostream& os) const {
  if (conflictMatrixInfo.allActions.empty()) {
    os << "  No conflict matrix information available\n";
    return;
  }
  
  const auto& actions = conflictMatrixInfo.allActions;
  size_t maxNameLength = 0;
  for (const auto& action : actions) {
    maxNameLength = std::max(maxNameLength, action.length());
  }
  maxNameLength = std::max(maxNameLength, size_t(8)); // Minimum width
  
  // Create table data
  std::vector<std::vector<std::string>> table;
  
  // Header row
  std::vector<std::string> headerRow;
  headerRow.push_back("Action");
  for (const auto& action : actions) {
    headerRow.push_back(action);
  }
  table.push_back(headerRow);
  
  // Data rows
  for (const auto& action1 : actions) {
    std::vector<std::string> row;
    row.push_back(action1);
    
    for (const auto& action2 : actions) {
      if (action1 == action2) {
        row.push_back("─"); // Self-conflict (diagonal)
      } else {
        std::string conflict = getConflictString(action1, action2);
        if (conflict.empty()) {
          conflict = getConflictString(action2, action1); // Try reverse
        }
        if (conflict.empty()) {
          row.push_back("?"); // Unknown
        } else {
          row.push_back(conflict);
        }
      }
    }
    table.push_back(row);
  }
  
  // Calculate column widths
  auto columnWidths = calculateColumnWidths(table);
  
  os << "+"; 
  printTableBorder(os, columnWidths);
  os << "+\n";
  
  os << "|";
  for (size_t i = 0; i < columnWidths.size(); ++i) {
    if (i == 0) {
      std::string headerTitle = "CONFLICT MATRIX";
      os << llvm::format(" %-*s ", (int)columnWidths[i], headerTitle.c_str());
    } else {
      os << std::string(columnWidths[i] + 2, ' ');
    }
    if (i < columnWidths.size() - 1) os << "|";
  }
  os << "|\n";
  
  os << "+";
  printTableBorder(os, columnWidths);
  os << "+\n";
  
  // Print header
  printTableRow(os, table[0], columnWidths);
  
  os << "+";
  printTableBorder(os, columnWidths);
  os << "+\n";
  
  // Print data rows
  for (size_t i = 1; i < table.size(); ++i) {
    printTableRow(os, table[i], columnWidths);
  }
  
  os << "+";
  printTableBorder(os, columnWidths);
  os << "+\n";
  
  // Print legend
  os << "\nLegend: SB=Sequence Before, SA=Sequence After, C=Conflict, CF=Conflict Free\n";
}

void ScheduleDebugger::printAll(llvm::raw_ostream& os) const {
  printModuleHeader(scheduleInfo.moduleName, os);
  os << "\n";
  printSchedule(os);
  os << "\n";
  printConflictMatrix(os);
  os << "\n";
}

void ScheduleDebugger::printModuleHeader(llvm::StringRef moduleName, llvm::raw_ostream& os) const {
  std::string header = "[" + passName + "] Module: " + moduleName.str() + " (" + timingMode + " mode)";
  std::string border(header.length() + 4, '=');
  
  os << "+" << border << "+\n";
  os << "|  " << header << "  |\n";
  os << "+" << border << "+\n";
}

std::string ScheduleDebugger::getConflictString(llvm::StringRef action1, llvm::StringRef action2) const {
  std::string key = action1.str() + "," + action2.str();
  auto it = conflictMatrixInfo.conflictEntries.find(key);
  if (it != conflictMatrixInfo.conflictEntries.end()) {
    return it->second;
  }
  return "";
}

void ScheduleDebugger::printTableBorder(llvm::raw_ostream& os, const std::vector<size_t>& columnWidths) const {
  for (size_t i = 0; i < columnWidths.size(); ++i) {
    for (size_t j = 0; j < columnWidths[i] + 2; ++j) {
      os << "-";
    }
    if (i < columnWidths.size() - 1) {
      os << "+";
    }
  }
}

void ScheduleDebugger::printTableRow(llvm::raw_ostream& os, const std::vector<std::string>& cells, 
                                     const std::vector<size_t>& columnWidths) const {
  os << "|";
  for (size_t i = 0; i < cells.size(); ++i) {
    if (i == 0) {
      // Left-align first column (action names)
      os << llvm::format(" %-*s ", (int)columnWidths[i], cells[i].c_str());
    } else {
      // Center-align conflict cells
      int padding = (int)columnWidths[i] - (int)cells[i].length();
      int leftPad = padding / 2;
      int rightPad = padding - leftPad;
      os << " " << std::string(leftPad, ' ') << cells[i] << std::string(rightPad, ' ') << " ";
    }
    if (i < cells.size() - 1) {
      os << "|";
    }
  }
  os << "|\n";
}

std::vector<size_t> ScheduleDebugger::calculateColumnWidths(const std::vector<std::vector<std::string>>& table) const {
  if (table.empty()) return {};
  
  std::vector<size_t> widths(table[0].size(), 0);
  
  for (const auto& row : table) {
    for (size_t i = 0; i < row.size() && i < widths.size(); ++i) {
      widths[i] = std::max(widths[i], row[i].length());
    }
  }
  
  // Ensure minimum width for readability
  for (auto& width : widths) {
    width = std::max(width, size_t(3));
  }
  
  return widths;
}

} // namespace sharp
} // namespace mlir