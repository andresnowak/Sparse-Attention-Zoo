#!/bin/bash
#SBATCH --job-name=llama-dsa-train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gpus-per-node=4

# ---- load .env ----
if [[ -f .env ]]; then
    # disable xtrace to avoid printing secrets
    export $(grep -v '^#' .env | xargs)
fi

set -x

ulimit -c 0 # In case the application crashes, it may leave behind large core dump files that contain an image of the process memory at the time of the crash. so we deactivate them if we don't need them for debugging

# Set environment variables
export WANDB_PROJECT=Sparse-Attention-Zoo
export NCCL_DEBUG=INFO
export TRANSFORMERS_VERBOSITY=info
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONUNBUFFERED=1
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export MASTER_PORT=6800

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200   # 2 hours (for dataset loading)

export head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

while [[ "$#" -gt 0 ]]; do
    arg="$1"
    case "$arg" in
        -c|--config)
            config_file="$2"
            shift 2
            ;;
        --checkpoint_path)
            checkpoint_path="$2"
            shift 2
            ;;
    esac
done

# Load config file if provided, otherwise use default args
if [ -n "$config_file" ]; then
    WARMUP_STAGE="" 
    BASELINE_EXPERIMENT=""

    CONFIG_FILE=$config_file
    echo "Using config file: $CONFIG_FILE"
    source $CONFIG_FILE
    WANDB_RUN_NAME="${WANDB_RUN_NAME}-${SLURM_JOB_ID}"
    SAVE_DIR="$SCRATCH/Sparse-Attention-Zoo/checkpoints/${WANDB_RUN_NAME}-${STAGE_NAME}"

else
    # Default configuration
    MODEL_NAME="meta-llama/Llama-3.2-1B"
    INDEX_TOP_K=2048
    INDEX_NUM_HEADS=16
    ROPE_HEAD_DIM=32
    INDEX_HEAD_DIM=64
    MICRO_BATCH_SIZE=4
    GLOBAL_BATCH_SIZE=16
    LEARNING_RATE=1e-4
    NUM_EPOCHS=3
    MAX_SEQ_LENGTH=2048
    DATASET_NAME="wikitext"
    DATASET_CONFIG="wikitext-2-raw-v1"
    DATASET_SPLIT="train"
    MAX_TRAIN_SAMPLES=10000
    DATASET_OFFSET=0
    WANDB_RUN_NAME="llama-dsa-${SLURM_JOB_ID}"
    SAVE_DIR="$SCRATCH/Sparse-Attention-Zoo/checkpoints/run-${SLURM_JOB_ID}"
    SAVE_EVERY=1000
    LOG_EVERY=10
    WEIGHT_DECAY=0.1
    GRADIENT_CLIPPING=inf
    WARMUP_STAGE=""  # Empty means not in warmup stage, set to "--warmup_stage" to enable
fi

# Print configuration
export MAX_TRAIN_SAMPLES=$((MAX_TRAIN_TOKENS / MAX_SEQ_LENGTH))
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL_NAME"
echo "Micro Batch size: $MICRO_BATCH_SIZE"
echo "Global Batch size: $GLOBAL_BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "Max train samples: $MAX_TRAIN_SAMPLES"
echo "Dataset: $DATASET_NAME/$DATASET_CONFIG"
echo "=========================================="


export LAUNCHER="accelerate launch \
    --config_file ./configs/ddp_config.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port $MASTER_PORT"

export SCRIPT="./main.py"
export SCRIPT_ARGS=" \
    --model_name $MODEL_NAME \
    --index_top_k $INDEX_TOP_K \
    --index_num_heads $INDEX_NUM_HEADS \
    --index_head_dim $INDEX_HEAD_DIM \
    --rope_head_dim $ROPE_HEAD_DIM \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --min_lr $MIN_LR \
    --num_epochs $NUM_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --dataset_name $DATASET_NAME \
    --dataset_config $DATASET_CONFIG \
    --dataset_split $DATASET_SPLIT \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --dataset_offset $DATASET_OFFSET \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --save_dir $SAVE_DIR \
    --save_every $SAVE_EVERY \
    --log_every $LOG_EVERY \
    --track_log_every $TRACK_LOG_EVERY \
    --weight_decay $WEIGHT_DECAY \
    --gradient_clipping $GRADIENT_CLIPPING \
    $WARMUP_STAGE \
    $BASELINE_EXPERIMENT \
    --track_token_selection
    "

if [ -n "$checkpoint_path" ]; then
  SCRIPT_ARGS="$SCRIPT_ARGS --load_from_checkpoint $checkpoint_path"
fi

srun --mpi=pmix --environment=pytorch2506 -u bash -lc '
set -x

echo "=== Node $SLURM_NODEID Debug Info ==="
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python before venv: $(which python3)"

source .venv/bin/activate

echo "Python after venv: $(which python3)"
echo "Accelerate location: $(which accelerate)"
echo "Master IP: $head_node_ip:$MASTER_PORT"
echo "Machine rank: $SLURM_NODEID"
echo "===================================="

'"$LAUNCHER"' --machine_rank $SLURM_NODEID '"$SCRIPT $SCRIPT_ARGS"'

echo "Training completed!"
'
