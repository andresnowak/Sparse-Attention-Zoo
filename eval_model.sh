#!/bin/bash
#SBATCH --job-name=llama-eval
#SBATCH --output=logs/evals/eval-%j.out
#SBATCH --error=logs/evals/eval-%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

# example: sbatch -A a139 eval_model.sh --model_path meta-llama/Llama-3.2-1B --model_type base 
# sbatch -A a139 eval_model.sh --model_path $SCRATCH/Sparse-Attention-Zoo/checkpoints/run-1089676-sparse/final_model --model_type dsa


set -x

# Default values
MODEL_PATH=""
MODEL_TYPE="base"
TASKS="hellaswag,arc_easy,winogrande,ruler,mmlu"
BATCH_SIZE="4"

# Parse arguments (support both named and positional)
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
    esac
done


if [ -z "$MODEL_PATH" ]; then
    echo "Usage: sbatch eval_model.sh --model_path <path> [--model_type <type>] [--tasks <tasks>] [--batch_size <size>]"
    echo "  Or positional: sbatch eval_model.sh <model_path> [model_type] [tasks] [batch_size]"
    echo ""
    echo "Options:"
    echo "  --model_path: Path to model (required)"
    echo "  --model_type: base or dsa (default: base)"
    echo "  --tasks: comma-separated list (default: hellaswag,arc_easy,winogrande,ruler)"
    echo "  --batch_size: evaluation batch size (default: 4)"
    exit 1
fi

echo "=========================================="
echo "MODEL EVALUATION"
echo "Model path: $MODEL_PATH"
echo "Model type: $MODEL_TYPE"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "=========================================="

# Set environment variables
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export MASTER_PORT=6801

# Output path based on job ID and model type
OUTPUT_PATH="evals/eval_results_${MODEL_TYPE}_${SLURM_JOB_ID}.json"

# Accelerate launcher for multi-GPU
export LAUNCHER="accelerate launch \
    --config_file ./configs/ddp_config.yaml \
    --num_processes $GPUS_PER_NODE \
    --num_machines 1 \
    --main_process_port $MASTER_PORT"

srun --mpi=pmix --environment=pytorch2506 -u bash -lc '
set -x
source .venv/bin/activate
'"$LAUNCHER"' eval_model.py \
    --model_path '"$MODEL_PATH"' \
    --model_type '"$MODEL_TYPE"' \
    --tasks '"$TASKS"' \
    --batch_size '"$BATCH_SIZE"' \
    --output_path '"$OUTPUT_PATH"'
'

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "✅ Evaluation completed successfully!"
echo "Results saved to: $OUTPUT_PATH"
