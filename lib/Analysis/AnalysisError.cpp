//===- AnalysisError.cpp - Structured Error Reporting Utility -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a utility class for creating structured error messages
// across Sharp analysis passes with consistent formatting.
//
//===----------------------------------------------------------------------===//

#include "sharp/Analysis/AnalysisError.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace sharp;

AnalysisError::AnalysisError(Operation *op, llvm::StringRef passName)
    : op(op), passName(passName) {}

AnalysisError& AnalysisError::setCategory(ErrorCategory category) {
  this->category = category;
  return *this;
}

AnalysisError& AnalysisError::setCategory(llvm::StringRef customCategory) {
  this->category = ErrorCategory::Custom;
  this->customCategory = customCategory;
  return *this;
}

AnalysisError& AnalysisError::setDetails(llvm::StringRef details) {
  this->details = details;
  return *this;
}

AnalysisError& AnalysisError::setReason(llvm::StringRef reason) {
  this->reason = reason;
  return *this;
}

AnalysisError& AnalysisError::setSolution(llvm::StringRef solution) {
  this->solution = solution;
  return *this;
}

llvm::StringRef AnalysisError::getCategoryString() const {
  switch (category) {
  case ErrorCategory::MissingDependency:
    return "missing dependency";
  case ErrorCategory::ValidationFailure:
    return "validation failed";
  case ErrorCategory::UnsupportedConstruct:
    return "unsupported construct";
  case ErrorCategory::InconsistentState:
    return "inconsistent state";
  case ErrorCategory::SchedulingFailure:
    return "scheduling failure";
  case ErrorCategory::InvalidContext:
    return "invalid context";
  case ErrorCategory::Custom:
    return customCategory;
  }
  return "unknown error";
}

mlir::InFlightDiagnostic AnalysisError::emit() {
  auto diag = op->emitError();
  diag << "[" << passName << "] Pass failed - " << getCategoryString();

  if (!details.empty()) {
    diag << ": " << details;
  }
  if (!reason.empty()) {
    diag << ". Reason: " << reason;
  }
  if (!solution.empty()) {
    diag << ". Solution: " << solution;
  }
  
  return diag;
}