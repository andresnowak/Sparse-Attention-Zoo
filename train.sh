#!/bin/bash
#SBATCH --job-name=llama-dsa-train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node

set -x

ulimit -c 0 # In case the application crashes, it may leave behind large core dump files that contain an image of the process memory at the time of the crash. so we deactivate them if we don't need them for debugging


# Set environment variables
export WANDB_PROJECT=Sparse-Attention-Zoo
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONUNBUFFERED=1
export GPUS_PER_NODE=4
export MASTER_PORT=6800

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Load config file if provided, otherwise use default args
if [ -n "$1" ]; then
    CONFIG_FILE=$1
    echo "Using config file: $CONFIG_FILE"
    source $CONFIG_FILE
    WANDB_RUN_NAME="${WANDB_RUN_NAME}-${SLURM_JOB_ID}"
    SAVE_DIR="$SCRATCH/Sparse-Attention-Zoo/checkpoints/run-${SLURM_JOB_ID}"

else
    # Default configuration
    MODEL_NAME="meta-llama/Llama-3.2-1B"
    INDEX_TOP_K=2048
    INDEX_NUM_HEADS=16
    ROPE_HEAD_DIM=32
    INDEX_HEAD_DIM=64
    BATCH_SIZE=4
    LEARNING_RATE=1e-4
    NUM_EPOCHS=3
    MAX_SEQ_LENGTH=2048
    GRADIENT_ACCUMULATION_STEPS=4
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
    WARMUP_STAGE=""  # Empty means not in warmup stage, set to "--warmup_stage" to enable
fi

# ---- load .env ----
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

# Print configuration
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Dataset: $DATASET_NAME/$DATASET_CONFIG"
echo "=========================================="


export LAUNCHER="accelerate launch \
    --config_file ./configs/fsdp_config.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port $MASTER_PORT \
    "
export SCRIPT="./main.py"
export SCRIPT_ARGS=" \
    --model_name $MODEL_NAME \
    --index_top_k $INDEX_TOP_K \
    --index_num_heads $INDEX_NUM_HEADS \
    --index_head_dim $INDEX_HEAD_DIM \
    --rope_head_dim $ROPE_HEAD_DIM \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
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
    --weight_decay $WEIGHT_DECAY \
    $WARMUP_STAGE
    "

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
srun 

srun --environment=pytorch2506 -u bash -lc '
set -x

source .venv/bin/activate

$CMD

echo "Training completed!"
'
