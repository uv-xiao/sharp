#!/bin/bash

# Script to translate a single MLIR file to FIRRTL and Verilog with specified timing mode
# Called by run.sh for each individual file translation

set -e # Exit immediately if a command exits with a non-zero status

# --- Configuration ---
SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"
FIRTOOL="$SHARP_ROOT/.install/unified/bin/firtool"

# --- Main Translation Function ---
#
# Translates a single MLIR file to FIRRTL and Verilog with specified timing mode
#
# @param $1: The input MLIR file (e.g., counter_hw.mlir)
# @param $2: The will-fire mode (static or dynamic)
# @param $3: Step number for display (optional)
#
translate_one() {
    local mlir_file=$1
    local mode=$2
    local step_num=${3:-""}
    local base_name=$(basename "${mlir_file}" .mlir)
    local fir_file="${base_name}_${mode}.fir"
    local v_file="${base_name}_${mode}.v"
    local log_file="${base_name}_${mode}_verilog.log"

    # Display header
    if [ -n "${step_num}" ]; then
        echo "${step_num}. Translating ${base_name} with ${mode} timing:"
        echo "$(printf '%*s' $((${#base_name} + ${#mode} + 30)) '' | tr ' ' '-')"
    else
        echo "Translating ${base_name} with ${mode} timing:"
        echo "$(printf '%*s' $((${#base_name} + ${#mode} + 20)) '' | tr ' ' '-')"
    fi

    # 1. Translate to FIRRTL (with debug output and two-pass conversion)
    echo "   - Generating FIRRTL..."
    local debug_file="${base_name}_${mode}_debug.log"
    local intermediate_file="${base_name}_${mode}_intermediate.mlir"
    
    # Enable conflict debugging and run with analysis passes
    export SHARP_DEBUG_CONFLICTS=1
    
    # Step 1: Lower Txn operation bodies to FIRRTL operations
    echo "     Step 1: Converting operation bodies to FIRRTL..."
    $SHARP_OPT "${mlir_file}" \
        --sharp-primitive-gen \
        --sharp-infer-conflict-matrix \
        --sharp-reachability-analysis \
        --sharp-general-check \
        --sharp-pre-synthesis-check \
        --lower-txn-body-to-firrtl \
        > "${intermediate_file}" 2>"${debug_file}"
    
    if [ $? -ne 0 ]; then
        echo "     ❌ Step 1 failed - operation body conversion failed"
        unset SHARP_DEBUG_CONFLICTS
        return 1
    fi
    
    # Print Step 1 results
    echo "     ✅ Step 1 completed - operation bodies converted to FIRRTL"
    echo "     📄 Intermediate IR (after Step 1, in ${intermediate_file}):"
    echo "     ╭─────────────────────────────────────────────────────────────────────────────────"
    if [ -f "${intermediate_file}" ] && [ -s "${intermediate_file}" ]; then
        head -20 "${intermediate_file}" | sed 's/^/     │ /'
        if [ $(wc -l < "${intermediate_file}") -gt 20 ]; then
            echo "     │ ... ($(wc -l < "${intermediate_file}") total lines)"
        fi
    else
        echo "     │ (empty or not generated)"
    fi
    echo "     ╰─────────────────────────────────────────────────────────────────────────────────"
    echo ""
    
    # Step 2: Translate Txn structure to FIRRTL modules with will-fire logic
    echo "     Step 2: Translating Txn structure to FIRRTL modules..."
    $SHARP_OPT "${intermediate_file}" \
        --translate-txn-to-firrtl=will-fire-mode="${mode}" \
        > "${fir_file}" 2>>"${debug_file}"
    
    unset SHARP_DEBUG_CONFLICTS
    
    # Print Step 2 results
    if [ $? -eq 0 ]; then
        echo "     ✅ Step 2 completed - Txn structure translated to FIRRTL"
        echo "     📄 Final FIRRTL IR (after Step 2):"
        echo "     ╭─────────────────────────────────────────────────────────────────────────────────"
        if [ -f "${fir_file}" ] && [ -s "${fir_file}" ]; then
            head -20 "${fir_file}" | sed 's/^/     │ /'
            if [ $(wc -l < "${fir_file}") -gt 20 ]; then
                echo "     │ ... ($(wc -l < "${fir_file}") total lines)"
            fi
        else
            echo "     │ (empty or not generated)"
        fi
        echo "     ╰─────────────────────────────────────────────────────────────────────────────────"
    else
        echo "     ❌ Step 2 failed - Txn structure translation failed"
        echo "     📄 Step 2 failed, but Step 1 intermediate results:"
        echo "     ╭─────────────────────────────────────────────────────────────────────────────────"
        if [ -f "${intermediate_file}" ] && [ -s "${intermediate_file}" ]; then
            head -20 "${intermediate_file}" | sed 's/^/     │ /'
            if [ $(wc -l < "${intermediate_file}") -gt 20 ]; then
                echo "     │ ... ($(wc -l < "${intermediate_file}") total lines)"
            fi
        else
            echo "     │ (empty or not generated)"
        fi
        echo "     ╰─────────────────────────────────────────────────────────────────────────────────"
        # Don't clean up intermediate file if Step 2 failed
        return 1
    fi
    
    # Clean up intermediate file only if both steps succeeded
    if [ $? -eq 0 ]; then
        echo "     ✅ FIRRTL generation successful ($(wc -l < "${fir_file}") lines)"
        # Analyze debug output for conflict resolution information
        if [ -f "${debug_file}" ] && [ -s "${debug_file}" ]; then
            conflict_debugs=$(grep -c "TxnToFIRRTL Debug" "${debug_file}" 2>/dev/null || echo "0")
            will_fire_decisions=$(grep -c "Will-Fire Generation" "${debug_file}" 2>/dev/null || echo "0")
            conflict_checks=$(grep -c "Conflict Detection" "${debug_file}" 2>/dev/null || echo "0")
            abort_conditions=$(grep -c "Abort Condition" "${debug_file}" 2>/dev/null || echo "0")
            echo "     📊 Debug info: ${will_fire_decisions} will-fire decisions, ${conflict_checks} conflict checks, ${abort_conditions} abort conditions"
        fi
    else
        echo "     ❌ FIRRTL generation failed"
        echo "     📁 Error details are in ${debug_file}"
        if [ -f "${debug_file}" ] && [ -s "${debug_file}" ]; then
            echo "     First few error lines:"
            head -5 "${debug_file}" | sed 's/^/       /'
            echo "     ..."
            echo "     📖 For complete error analysis, check: ${debug_file}"
        fi
        return 1
    fi

    # 2. Translate to Verilog
    echo "   - Generating Verilog..."
    
    # Use timeout with proper signal handling
    if timeout --signal=KILL 30 $FIRTOOL "${fir_file}" --format=mlir --verilog -o "${v_file}" 2>"${log_file}"; then
        if [ -f "${v_file}" ] && [ -s "${v_file}" ]; then
            echo "     ✅ Verilog generation successful ($(wc -l < "${v_file}") lines)"
            echo "     📁 Output: ${fir_file}, ${v_file}"
            
            # Show module interface for verification
            echo "     Module interface:"
            grep -E "(module|input|output)" "${v_file}" | head -5 | sed 's/^/       /'
            rm "${log_file}" 2>/dev/null || true
        else
            echo "     ❌ Verilog generation failed (empty output)"
            echo "     📁 Output: ${fir_file} (FIRRTL only)"
            if [ -f "${log_file}" ]; then
                error_count=$(grep -c "error:" "${log_file}" 2>/dev/null || echo "0")
                echo "     Found ${error_count} synthesis errors"
                rm "${log_file}"
            fi
        fi
    else
        exit_code=$?
        if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
            echo "     ❌ Verilog generation timed out (30s limit exceeded)"
        else
            echo "     ❌ Verilog generation failed"
        fi
        echo "     📁 Output: ${fir_file} (FIRRTL only)"
        
        # Show error details
        if [ -f "${log_file}" ] && [ -s "${log_file}" ]; then
            error_count=$(grep -c "error:" "${log_file}" 2>/dev/null || echo "0")
            echo "     Found ${error_count} synthesis errors in firtool"
            echo "     First few errors:"
            head -3 "${log_file}" | sed 's/^/       /'
            rm "${log_file}"
        fi
    fi
    echo ""
}

# --- Main Execution Logic ---

# Validate arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <mlir_file> <mode> [step_number]"
    echo "  mlir_file: Input MLIR file to translate"
    echo "  mode: Timing mode (static or dynamic)"
    echo "  step_number: Optional step number for display"
    echo ""
    echo "Example: $0 counter_hw.mlir dynamic 1"
    exit 1
fi

# Extract arguments
MLIR_FILE=$1
MODE=$2
STEP_NUM=${3:-""}

# Validate inputs
if [ ! -f "${MLIR_FILE}" ]; then
    echo "Error: Input file '${MLIR_FILE}' not found"
    exit 1
fi

if [[ ! "${MODE}" =~ ^(static|dynamic)$ ]]; then
    echo "Error: Invalid mode '${MODE}'. Must be 'static' or 'dynamic'"
    exit 1
fi

# Run the translation
translate_one "${MLIR_FILE}" "${MODE}" "${STEP_NUM}"