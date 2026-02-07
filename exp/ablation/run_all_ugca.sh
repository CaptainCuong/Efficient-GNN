#!/bin/bash
# Run all UGCA attack scripts for all datasets with a specified calibration method
# Skips experiments that already have logs in the logs directory
#
# Usage: ./run_all_ugca.sh <calibration_method> [additional_args]
#
# Example:
#   ./run_all_ugca.sh TS
#   ./run_all_ugca.sh CaGCN --budget 10
#   ./run_all_ugca.sh GETS --gets-experts 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root so ./logs is consistent with where Python scripts save
cd "$PROJECT_ROOT"

LOG_DIR="./logs"

CALIBRATION_METHODS=("CaGCN" "DCGC" "ETS" "GATS" "GETS" "MS" "TS" "VS" "WATS" "SimCalib")
DATASETS=("Cora" "CiteSeer" "PubMed" "CoraML" "Ogbn-arxiv" "Photo" "Physics" "Reddit")
UGCA_SCRIPTS=(
    "ugca_full_multi_dataset.py"
    "ugca_rerank_basic_multi_dataset.py"
    "ugca_rerank_hybridloss_multi_dataset.py"
    "ugca_under_kl_multi_dataset.py"
    "ugca_under_multi_dataset.py"
)

# Extract log prefix from script name (e.g., ugca_full_multi_dataset.py -> ugca_full)
get_log_prefix() {
    local script="$1"
    echo "$script" | sed 's/_multi_dataset\.py$//'
}

# Check if experiment log already exists
# Log pattern: {prefix}_{dataset}_{calibration_method}_attack_log_*.txt
experiment_exists() {
    local prefix="$1"
    local dataset="$2"
    local calib_method="$3"

    local dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    local calib_lower=$(echo "$calib_method" | tr '[:upper:]' '[:lower:]')
    local pattern="${LOG_DIR}/${prefix}_${dataset_lower}_${calib_lower}_*.txt"

    # Check if any matching file exists
    if ls $pattern 1> /dev/null 2>&1; then
        return 0  # exists
    else
        return 1  # does not exist
    fi
}

usage() {
    echo "Usage: $0 <calibration_method> [additional_args]"
    echo ""
    echo "Calibration methods: ${CALIBRATION_METHODS[*]}"
    echo "Datasets: ${DATASETS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0 TS"
    echo "  $0 CaGCN --budget 10"
    echo "  $0 GETS --gets-experts 5"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

CALIB_METHOD="$1"
shift
EXTRA_ARGS="$@"

valid_method=false
for method in "${CALIBRATION_METHODS[@]}"; do
    if [ "$CALIB_METHOD" == "$method" ]; then
        valid_method=true
        break
    fi
done

if [ "$valid_method" == "false" ]; then
    echo "Error: Invalid calibration method '$CALIB_METHOD'"
    echo "Valid methods: ${CALIBRATION_METHODS[*]}"
    exit 1
fi

echo "=========================================="
echo "Running all UGCA attacks"
echo "Calibration method: $CALIB_METHOD"
echo "Additional args: $EXTRA_ARGS"
echo "Log directory: $LOG_DIR"
echo "=========================================="
echo ""

total_runs=$((${#UGCA_SCRIPTS[@]} * ${#DATASETS[@]}))
current_run=0
skipped_runs=0
failed_runs=()

for script in "${UGCA_SCRIPTS[@]}"; do
    log_prefix=$(get_log_prefix "$script")

    for dataset in "${DATASETS[@]}"; do
        current_run=$((current_run + 1))

        # Skip CaGCN for Ogbn-arxiv and Reddit (not applicable)
        if [ "$CALIB_METHOD" == "CaGCN" ] && { [ "$dataset" == "Reddit" ]; }; then
            echo "[$current_run/$total_runs] SKIPPED: $script on $dataset with $CALIB_METHOD (not applicable)"
            skipped_runs=$((skipped_runs + 1))
            continue
        fi

        # Check if experiment already exists
        if experiment_exists "$log_prefix" "$dataset" "$CALIB_METHOD"; then
            echo "[$current_run/$total_runs] SKIPPED: $script on $dataset with $CALIB_METHOD (log already exists)"
            skipped_runs=$((skipped_runs + 1))
            continue
        fi

        echo "[$current_run/$total_runs] Running $script on $dataset with $CALIB_METHOD..."
        echo "Command: python $SCRIPT_DIR/$script --dataset $dataset --calibration-method $CALIB_METHOD $EXTRA_ARGS"
        echo "------------------------------------------"

        python "$SCRIPT_DIR/$script" \
            --dataset "$dataset" \
            --calibration-method "$CALIB_METHOD" \
            $EXTRA_ARGS

        if [ $? -ne 0 ]; then
            echo "WARNING: $script failed on $dataset"
            failed_runs+=("$script on $dataset")
        fi

        echo ""
    done
done

echo "=========================================="
echo "All runs completed!"
echo "Total: $total_runs | Skipped: $skipped_runs | Ran: $((total_runs - skipped_runs))"
echo "=========================================="

if [ ${#failed_runs[@]} -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    for run in "${failed_runs[@]}"; do
        echo "  - $run"
    done
    exit 1
fi
