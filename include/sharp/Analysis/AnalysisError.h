#ifndef SHARP_ANALYSIS_ANALYSISERROR_H
#define SHARP_ANALYSIS_ANALYSISERROR_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sharp {

/// Enum for different error categories
enum class ErrorCategory {
  MissingDependency,
  ValidationFailure,
  UnsupportedConstruct,
  InconsistentState,
  SchedulingFailure,
  InvalidContext,
  Custom
};

/// Utility class for creating structured error messages across Sharp analysis passes
/// 
/// Usage:
///   AnalysisError(op, "PassName")
///     .setCategory(ErrorCategory::MissingDependency)
///     .setDetails("detailed description")
///     .setReason("why this is a problem")
///     .setSolution("how to fix it")
///     .emit();
class AnalysisError {
public:
  AnalysisError(Operation *op, llvm::StringRef passName);

  /// Set error category using predefined types
  AnalysisError& setCategory(ErrorCategory category);
  
  /// Set custom error category string
  AnalysisError& setCategory(llvm::StringRef customCategory);
  
  /// Set detailed description of the issue
  AnalysisError& setDetails(llvm::StringRef details);
  
  /// Set explanation of why this is a problem
  AnalysisError& setReason(llvm::StringRef reason);
  
  /// Set actionable advice on how to fix the issue
  AnalysisError& setSolution(llvm::StringRef solution);

  /// Emit the structured error message and return InFlightDiagnostic for chaining
  mlir::InFlightDiagnostic emit();

private:
  Operation *op;
  llvm::StringRef passName;
  ErrorCategory category = ErrorCategory::Custom;
  llvm::StringRef customCategory;
  llvm::StringRef details;
  llvm::StringRef reason;
  llvm::StringRef solution;

  llvm::StringRef getCategoryString() const;
};

} // namespace sharp
} // namespace mlir

#endif // SHARP_ANALYSIS_ANALYSISERROR_H