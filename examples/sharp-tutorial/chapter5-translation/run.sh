#!/bin/bash

# Orchestrates translation of multiple MLIR files by calling translate_one.sh
# This script handles batch processing and analysis summaries

set -e # Exit immediately if a command exits with a non-zero status

# --- Configuration ---
SCRIPT_DIR="$(dirname "$0")"
TRANSLATE_ONE="$SCRIPT_DIR/translate_one.sh"

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
    local step_num=$3
    
    # Call translate_one.sh to handle the actual translation
    "$TRANSLATE_ONE" "${mlir_file}" "${mode}" "${step_num}"
    return $?
}

# --- Analysis Function ---
#
# Analyzes the generated FIRRTL for timing characteristics
#
# @param $1: The FIRRTL file to analyze
# @param $2: The timing mode
#
analyze_firrtl() {
    local fir_file=$1
    local mode=$2

    if [ -f "${fir_file}" ]; then
        echo "   Analysis for ${mode} mode:"
        echo "   - Will-fire signals: $(grep -c "_wf" "${fir_file}")"
        echo "   - Ready signals: $(grep -c "RDY" "${fir_file}")"
        echo "   - Enable signals: $(grep -c "EN" "${fir_file}")"
        echo "   - When blocks: $(grep -c "firrtl.when" "${fir_file}")"
        echo ""
    fi
}

# --- Debug Analysis Function ---
#
# Analyzes debug logs for conflict resolution insights
#
# @param $1: The debug log file to analyze
# @param $2: The timing mode
#
analyze_debug_log() {
    local debug_file=$1
    local mode=$2
    local base_name=$(basename "${debug_file}" "_${mode}_debug.log")

    if [ -f "${debug_file}" ] && [ -s "${debug_file}" ]; then
        echo "   Debug Analysis for ${base_name} (${mode}):"
        
        # Count different types of debug information
        local will_fire_count=$(grep -c "Will-Fire Generation" "${debug_file}" 2>/dev/null || echo "0")
        local conflict_count=$(grep -c "Conflict Detection" "${debug_file}" 2>/dev/null || echo "0") 
        local abort_count=$(grep -c "Abort Condition" "${debug_file}" 2>/dev/null || echo "0")
        local reachability_count=$(grep -c "Reachability Analysis" "${debug_file}" 2>/dev/null || echo "0")
        
        echo "     - Will-fire decisions: ${will_fire_count}"
        echo "     - Conflict detections: ${conflict_count}"
        echo "     - Abort conditions: ${abort_count}"
        echo "     - Reachability analyses: ${reachability_count}"
        
        # Extract key decision information
        local success_count=$(grep -c "SUCCESS:" "${debug_file}" 2>/dev/null | head -1 || echo "0")
        local failure_count=$(grep -c "FAILED:" "${debug_file}" 2>/dev/null | head -1 || echo "0")
        
        if [ "${success_count}" -gt 0 ] || [ "${failure_count}" -gt 0 ]; then
            echo "     - Successful operations: ${success_count}"
            if [ "${failure_count}" -gt 0 ]; then
                echo "     - Failed operations: ${failure_count}"
            fi
        fi
        
        # Show any error patterns
        local conflict_blocks=$(grep -c "CONFLICT" "${debug_file}" 2>/dev/null | head -1 || echo "0")
        if [ "${conflict_blocks}" -gt 0 ]; then
            echo "     - Conflicts detected: ${conflict_blocks}"
        fi
        
        echo ""
    fi
}

# --- Main Execution Logic ---

# Define the MLIR files to translate
MLIR_FILES=("counter_hw.mlir" "nested_modules.mlir" "datapath.mlir" "conditional_logic.mlir")

# Check if mode is provided as argument
if [ $# -eq 1 ]; then
    MODE=$1
    if [[ ! "${MODE}" =~ ^(static|dynamic)$ ]]; then
        echo "Error: Invalid mode '${MODE}'. Must be 'static' or 'dynamic'"
        exit 1
    fi
    
    echo "=== Chapter 5: Translation with ${MODE^} Timing Mode ==="
    echo ""
    echo "${MODE^} Timing Mode:"
    echo "- ${MODE_DESCRIPTIONS[$MODE]}"
    echo ""
    
    # Run single mode for all files
    step=1
    for file in "${MLIR_FILES[@]}"; do
        if [ -f "${file}" ]; then
            run_translation "${file}" "${MODE}" "${step}"
            ((step++))
        else
            echo "Warning: File ${file} not found, skipping..."
        fi
    done
    
    # Show analysis summary
    echo "Analysis Summary for ${MODE^} Mode:"
    echo "$(printf '%*s' 40 '' | tr ' ' '=')"
    for file in "${MLIR_FILES[@]}"; do
        base_name=$(basename "${file}" .mlir)
        fir_file="${base_name}_${MODE}.fir"
        debug_file="${base_name}_${MODE}_debug.log"
        analyze_firrtl "${fir_file}" "${MODE}"
        analyze_debug_log "${debug_file}" "${MODE}"
    done
    
elif [ $# -eq 0 ]; then
    # Run all modes
    MODES=("static" "dynamic")
    
    for mode in "${MODES[@]}"; do
        echo "=== Chapter 5: Translation with ${mode^} Timing Mode ==="
        echo ""
        echo "${mode^} Timing Mode:"
        echo "- ${MODE_DESCRIPTIONS[$mode]}"
        echo ""
        
        step=1
        for file in "${MLIR_FILES[@]}"; do
            if [ -f "${file}" ]; then
                run_translation "${file}" "${mode}" "${step}"
                ((step++))
            else
                echo "Warning: File ${file} not found, skipping..."
            fi
        done
        
        # Show analysis summary
        echo "Analysis Summary for ${mode^} Mode:"
        echo "$(printf '%*s' 40 '' | tr ' ' '=')"
        for file in "${MLIR_FILES[@]}"; do
            base_name=$(basename "${file}" .mlir)
            fir_file="${base_name}_${mode}.fir"
            debug_file="${base_name}_${mode}_debug.log"
            analyze_firrtl "${fir_file}" "${mode}"
            analyze_debug_log "${debug_file}" "${mode}"
        done
        echo ""
    done
    
    echo "🎉 All translations completed successfully for all timing modes."
else
    echo "Usage: $0 [mode]"
    echo "  mode: static or dynamic (optional)"
    echo "  If no mode specified, runs all three modes"
    exit 1
fi