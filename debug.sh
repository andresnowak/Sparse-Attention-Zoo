#!/bin/bash
# filepath: /users/anowak/developer/Sparse-Attention-Zoo/debug.sh
# Debug script for local development with debugpy

# Set environment variables
export WANDB_PROJECT=Sparse-Attention-Zoo
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# Default configuration
MODEL_NAME="meta-llama/Llama-3.2-1B"
INDEX_TOP_K=8
INDEX_NUM_HEADS=16
ROPE_HEAD_DIM=32
INDEX_HEAD_DIM=64
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1
LEARNING_RATE=1e-4
MIN_LR=1e-5
NUM_EPOCHS=1
MAX_SEQ_LENGTH=32
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1" 
DATASET_SPLIT="train"
MAX_TRAIN_SAMPLES=100
DATASET_OFFSET=0
WANDB_RUN_NAME="debug-run"
SAVE_DIR="./checkpoints/debug"
SAVE_EVERY=100
LOG_EVERY=10
TRACK_LOG_EVERY=10
WEIGHT_DECAY=0.1
GRADIENT_CLIPPING=1.0
WARMUP_STAGE=""
BASELINE_EXPERIMENT=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -c|--config)
            config_file="$2"
            shift 2
            ;;
        --checkpoint_path)
            checkpoint_path="$2"
            shift 2
            ;;
        --port)
            DEBUG_PORT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Load config file if provided
if [ -n "$config_file" ]; then
    echo "Using config file: $config_file"
    source $config_file
fi

# Set debug port (default: 5678)
DEBUG_PORT=${DEBUG_PORT:-5678}

echo "=========================================="
echo "DEBUG MODE"
echo "Debug port: $DEBUG_PORT"
echo "Model: $MODEL_NAME"
echo "Micro Batch size: $MICRO_BATCH_SIZE"
echo "Global Batch size: $GLOBAL_BATCH_SIZE"
echo "Max train samples: $MAX_TRAIN_SAMPLES"
echo "=========================================="

SCRIPT_ARGS=" \
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

IP_HOST=$SLURMD_NODENAME
echo "Starting debugpy on port $DEBUG_PORT..."
python -m debugpy --listen $IP_HOST:$DEBUG_PORT --wait-for-client ./main.py $SCRIPT_ARGS