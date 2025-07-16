#!/bin/bash

# Orchestrates translation of multiple MLIR files by calling translate_one.sh
# This script handles batch processing and analysis summaries

set -e # Exit immediately if a command exits with a non-zero status

# --- Configuration ---
SCRIPT_DIR="$(dirname "$0")"
TRANSLATE_ONE="$SCRIPT_DIR/translate_one.sh"

until=$1

# Ensure translate_one.sh exists and is executable
if [ ! -f "$TRANSLATE_ONE" ]; then
    echo "Error: translate_one.sh not found at $TRANSLATE_ONE"
    exit 1
fi

if [ ! -x "$TRANSLATE_ONE" ]; then
    chmod +x "$TRANSLATE_ONE"
fi

# --- Mode descriptions ---
declare -A MODE_DESCRIPTIONS=(
    ["static"]="Conservative will-fire logic generation. All actions conservatively marked as conflicting unless proven safe. Minimal hardware optimization but maximum correctness."
    ["dynamic"]="Precise will-fire logic generation using actual dependencies. Analyzes method call patterns to optimize conflict detection. Balanced between hardware efficiency and analysis complexity."
    # Removed most-dynamic mode - it's equivalent to dynamic with proper abort handling
)

# --- Translation Orchestration Function ---
#
# Calls translate_one.sh for each file and timing mode
#
# @param $1: The input MLIR file (e.g., counter_hw.mlir)
# @param $2: The will-fire mode (static or dynamic)
# @param $3: Step number for display
#
run_translation() {
    local mlir_file=$1
    local mode=$2
    local until=$3
    
    # Call translate_one.sh to handle the actual translation
    "$TRANSLATE_ONE" "${mlir_file}" "${mode}" "${until}"
    return $?
}


# --- Main Execution Logic ---

# Define the MLIR files to translate
MLIR_FILES=("counter_hw.mlir" "nested_modules.mlir" "datapath.mlir" "conditional_logic.mlir")
    
if [ $# -eq 1 ]; then
    # Run all modes
    MODES=("static" "dynamic")
    
    for mode in "${MODES[@]}"; do
        echo "=== Chapter 5: Translation with ${mode^} Timing Mode ==="
        echo ""
        echo "${mode^} Timing Mode:"
        echo "- ${MODE_DESCRIPTIONS[$mode]}"
        echo ""
        
        for file in "${MLIR_FILES[@]}"; do
            if [ -f "${file}" ]; then
                run_translation "${file}" "${mode}" "${UNTIL}"
            else
                echo "Warning: File ${file} not found, skipping..."
            fi
        done

    done
    
    echo "🎉 All translations completed successfully for all timing modes."
else
    echo "Usage: $0 [until]"
    echo "  until: analysis, firrtl-op, firrtl, verilog"
    echo "  If no until specified, runs all steps"
    exit 1
fi