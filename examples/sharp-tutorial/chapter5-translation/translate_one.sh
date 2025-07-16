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
    local until=$3

    local base_name=$(basename "${mlir_file}" .mlir)
    local fir_file="${base_name}_${mode}.fir"
    local v_file="${base_name}_${mode}.v"
    local log_file="${base_name}_${mode}_verilog.log"

    export SHARP_DEBUG_CONFLICTS=1

    # Display header

    echo "Translating ${base_name} with ${mode} will-fire mode:"
    echo "$(printf '%*s' $((${#base_name} + ${#mode} + 20)) '' | tr ' ' '-')"

    # Step 1: do dependent analysis
    echo "  Step 1: do dependent analysis"
    local debug_file="${base_name}_analysis.log"
    local analysis_file="${base_name}_analysis.mlir"

    $SHARP_OPT "${mlir_file}" \
        --sharp-primitive-gen \
        --sharp-infer-conflict-matrix \
        --sharp-reachability-analysis \
        --sharp-general-check \
        --sharp-pre-synthesis-check \
        > "${analysis_file}" 2>"${debug_file}"
    
    if [ $? -ne 0 ]; then
        echo "     ❌ Step 1 failed - analysis failed"
        unset SHARP_DEBUG_CONFLICTS
        return 1
    else
        echo "     ✅ Step 1 completed - analysis completed"
    fi

    if [ "${until}" == "analysis" ]; then
        return 0
    fi

    # Step 2: Lower Txn operation bodies to FIRRTL operations
    echo "  Step 2: Lower Txn operation bodies to FIRRTL operations"
    local debug_file="${base_name}_firrtl.log"
    local firrtl_op_file="${base_name}_firrtl_op.mlir"

    $SHARP_OPT "${analysis_file}" \
        --lower-op-to-firrtl \
        > "${firrtl_op_file}" 2>"${debug_file}"
    
    if [ $? -ne 0 ]; then
        echo "     ❌ Step 2 failed - operation body conversion failed"
        unset SHARP_DEBUG_CONFLICTS
        return 1
    else
        echo "     ✅ Step 2 completed - operation bodies converted to FIRRTL"
        # Print Step 1 results
        echo "     📄 Intermediate IR (after LowerOpToFIRRTLPass, in ${firrtl_op_file}):"
        echo "     ╭─────────────────────────────────────────────────────────────────────────────────"
        if [ -f "${firrtl_op_file}" ] && [ -s "${firrtl_op_file}" ]; then
            head -20 "${firrtl_op_file}" | sed 's/^/     │ /'
            if [ $(wc -l < "${firrtl_op_file}") -gt 20 ]; then
                echo "     │ ... ($(wc -l < "${firrtl_op_file}") total lines)"
            fi
        else
            echo "     │ (empty or not generated)"
        fi
        echo "     ╰─────────────────────────────────────────────────────────────────────────────────"
        echo ""
    fi

    if [ "${until}" == "firrtl-op" ]; then
        return 0
    fi

    # Step 3: Translate to FIRRTL (with debug output and two-pass conversion)
    echo "  Step 3: Translate to FIRRTL"
    local debug_file="${base_name}_${mode}.log"
    local firrtl_file="${base_name}_${mode}.firrtl"
    
    $SHARP_OPT "${firrtl_op_file}" \
        --translate-txn-to-firrtl=will-fire-mode="${mode}" \
        > "${firrtl_file}" 2>"${debug_file}"

    if [ $? -ne 0 ]; then
        echo "     ❌ Step 3 failed - FIRRTL generation failed"
        unset SHARP_DEBUG_CONFLICTS
        return 1
    else
        echo "     ✅ Step 3 completed - FIRRTL generation successful"
        # Print Step 3 results
        echo "     📄 Final FIRRTL IR (after Step 3):"
        echo "     ╭─────────────────────────────────────────────────────────────────────────────────"
        if [ -f "${firrtl_file}" ] && [ -s "${firrtl_file}" ]; then
            head -20 "${firrtl_file}" | sed 's/^/     │ /'
            if [ $(wc -l < "${firrtl_file}") -gt 20 ]; then
                echo "     │ ... ($(wc -l < "${firrtl_file}") total lines)"
            fi
        else
            echo "     │ (empty or not generated)"
        fi
        echo "     ╰─────────────────────────────────────────────────────────────────────────────────"
        echo ""
    fi

    if [ "${until}" == "firrtl" ]; then
        return 0
    fi

    # Step 4: Translate to Verilog
    echo "  Step 4: Translate to Verilog"
    local verilog_log="${base_name}_${mode}_verilog.log"
    local v_file="${base_name}_${mode}.v"

    # Use timeout with proper signal handling
    if timeout --signal=KILL 30 $FIRTOOL "${firrtl_file}" --format=mlir --verilog -o "${v_file}" 2>"${verilog_log}"; then
        if [ -f "${v_file}" ] && [ -s "${v_file}" ]; then
            echo "     ✅ Verilog generation successful ($(wc -l < "${v_file}") lines)"
            echo "     📁 Output: ${firrtl_file}, ${v_file}"
            
            # Show module interface for verification
            echo "     Module interface:"
            grep -E "(module|input|output)" "${v_file}" | head -5 | sed 's/^/       /'
            rm "${verilog_log}" 2>/dev/null || true
        else
            echo "     ❌ Verilog generation failed (empty output)"
            echo "     📁 Output: ${firrtl_file} (FIRRTL only)"
            if [ -f "${verilog_log}" ]; then
                error_count=$(grep -c "error:" "${verilog_log}" 2>/dev/null || echo "0")
                echo "     Found ${error_count} synthesis errors"
                rm "${verilog_log}"
            fi
        fi
    else
        exit_code=$?
        if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
            echo "     ❌ Verilog generation timed out (30s limit exceeded)"
        else
            echo "     ❌ Verilog generation failed"
        fi
        echo "     📁 Output: ${firrtl_file} (FIRRTL only)"
        
        # Show error details
        if [ -f "${verilog_log}" ] && [ -s "${verilog_log}" ]; then
            error_count=$(grep -c "error:" "${verilog_log}" 2>/dev/null || echo "0")
            echo "     Found ${error_count} synthesis errors in firtool"
            echo "     First few errors:"
            head -3 "${verilog_log}" | sed 's/^/       /'
            rm "${verilog_log}"
        fi
    fi
    echo ""
}

# --- Main Execution Logic ---

# Validate arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <mlir_file> <mode>"
    echo "  mlir_file: Input MLIR file to translate"
    echo "  mode: Timing mode (static or dynamic)"
    echo "  until: Which step to stop (analysis, firrtl-op, firrtl, verilog)"
    echo ""
    echo "Example: $0 counter_hw.mlir dynamic"
    exit 1
fi

# Extract arguments
MLIR_FILE=$1
MODE=$2
UNTIL=$3

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
translate_one "${MLIR_FILE}" "${MODE}" "${UNTIL}"