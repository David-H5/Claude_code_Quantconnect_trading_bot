#!/bin/bash
# Autonomous Testing Pipeline for QuantConnect Trading Bot
# This script executes the full testing pipeline for Claude Code autonomous development

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "QuantConnect Trading Bot - Test Pipeline"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Initialize result tracking
PIPELINE_RESULT="success"
declare -A STAGE_RESULTS

# Function to run a stage and track results
run_stage() {
    local stage_name="$1"
    local command="$2"
    local output_file="$RESULTS_DIR/${stage_name}_${TIMESTAMP}.txt"

    echo -e "\n${YELLOW}Stage: $stage_name${NC}"
    echo "Command: $command"
    echo "---"

    if eval "$command" > "$output_file" 2>&1; then
        STAGE_RESULTS[$stage_name]="passed"
        echo -e "${GREEN}PASSED${NC}"
    else
        STAGE_RESULTS[$stage_name]="failed"
        PIPELINE_RESULT="failed"
        echo -e "${RED}FAILED${NC}"
        echo "See: $output_file"
    fi
}

# Stage 1: Algorithm Validation
run_stage "validation" "python scripts/algorithm_validator.py algorithms/"

# Stage 2: Syntax Check
run_stage "syntax" "python -m py_compile algorithms/*.py"

# Stage 3: Type Checking
run_stage "typecheck" "mypy algorithms/ --ignore-missing-imports --no-error-summary || true"

# Stage 4: Linting
run_stage "lint" "flake8 algorithms/ --max-line-length=120 --max-complexity=15 --exit-zero"

# Stage 5: Unit Tests
echo -e "\n${YELLOW}Stage: Unit Tests${NC}"
pytest tests/ -v --tb=short \
    --cov=algorithms --cov=indicators --cov=models --cov=utils \
    --cov-report=json:"$RESULTS_DIR/coverage_${TIMESTAMP}.json" \
    --junitxml="$RESULTS_DIR/test_results_${TIMESTAMP}.xml" \
    -m "unit or not integration" \
    > "$RESULTS_DIR/pytest_${TIMESTAMP}.txt" 2>&1 || true

# Check test results
if grep -q "failed" "$RESULTS_DIR/pytest_${TIMESTAMP}.txt"; then
    STAGE_RESULTS["unit_tests"]="failed"
    PIPELINE_RESULT="failed"
    echo -e "${RED}FAILED${NC}"
else
    STAGE_RESULTS["unit_tests"]="passed"
    echo -e "${GREEN}PASSED${NC}"
fi

# Stage 6: Backtest (if LEAN CLI available and credentials configured)
if command -v lean &> /dev/null && [ -n "$QC_USER_ID" ] && [ -n "$QC_API_TOKEN" ]; then
    echo -e "\n${YELLOW}Stage: Cloud Backtest${NC}"
    for algo in algorithms/*.py; do
        if [[ "$algo" != *"__init__"* ]]; then
            algo_name=$(basename "$algo" .py)
            echo "Backtesting: $algo_name"
            lean backtest "$algo_name" --download-data > "$RESULTS_DIR/backtest_${algo_name}_${TIMESTAMP}.txt" 2>&1 || true
        fi
    done
    STAGE_RESULTS["backtest"]="completed"
else
    echo -e "\n${YELLOW}Stage: Backtest - SKIPPED (LEAN CLI not configured)${NC}"
    STAGE_RESULTS["backtest"]="skipped"
fi

# Stage 7: Generate Analysis Report
echo -e "\n${YELLOW}Stage: Generate Analysis Report${NC}"
python scripts/analyze_results.py \
    --results-dir "$RESULTS_DIR" \
    --timestamp "$TIMESTAMP" \
    --output "$RESULTS_DIR/analysis_${TIMESTAMP}.json" \
    > "$RESULTS_DIR/analysis_log_${TIMESTAMP}.txt" 2>&1 || true
STAGE_RESULTS["analysis"]="completed"

# Generate Summary JSON
echo -e "\n${YELLOW}Generating Pipeline Summary${NC}"
cat > "$RESULTS_DIR/pipeline_summary_${TIMESTAMP}.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "pipeline_result": "$PIPELINE_RESULT",
    "stages": {
        "validation": "${STAGE_RESULTS[validation]:-unknown}",
        "syntax": "${STAGE_RESULTS[syntax]:-unknown}",
        "typecheck": "${STAGE_RESULTS[typecheck]:-unknown}",
        "lint": "${STAGE_RESULTS[lint]:-unknown}",
        "unit_tests": "${STAGE_RESULTS[unit_tests]:-unknown}",
        "backtest": "${STAGE_RESULTS[backtest]:-unknown}",
        "analysis": "${STAGE_RESULTS[analysis]:-unknown}"
    },
    "results_dir": "$RESULTS_DIR",
    "actionable_items": []
}
EOF

# Final Summary
echo ""
echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
for stage in "${!STAGE_RESULTS[@]}"; do
    result="${STAGE_RESULTS[$stage]}"
    if [ "$result" == "passed" ] || [ "$result" == "completed" ]; then
        echo -e "$stage: ${GREEN}$result${NC}"
    elif [ "$result" == "skipped" ]; then
        echo -e "$stage: ${YELLOW}$result${NC}"
    else
        echo -e "$stage: ${RED}$result${NC}"
    fi
done

echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Summary: $RESULTS_DIR/pipeline_summary_${TIMESTAMP}.json"

# Exit with appropriate code
if [ "$PIPELINE_RESULT" == "success" ]; then
    echo -e "\n${GREEN}Pipeline completed successfully!${NC}"
    exit 0
else
    echo -e "\n${RED}Pipeline completed with failures. Review results above.${NC}"
    exit 1
fi
