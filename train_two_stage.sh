#!/bin/bash
#SBATCH --job-name=llama-dsa-two-stage
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

# Load .env
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

set -x

ulimit -c 0

# Two-stage training: warmup then sparse
WARMUP_CONFIG="${1:-configs/warmup_stage.conf}"
SPARSE_CONFIG="${2:-configs/sparse_stage.conf}"

echo "=========================================="
echo "TWO-STAGE TRAINING MODE"
echo "Warmup config: $WARMUP_CONFIG"
echo "Sparse config: $SPARSE_CONFIG"
echo "=========================================="

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

export LAUNCHER="accelerate launch \
    --config_file ./configs/ddp_config.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port $MASTER_PORT"
export SCRIPT="./main.py"

# ========================================
# STAGE 1: WARMUP
# ========================================
echo "=========================================="
echo "STAGE 1: WARMUP TRAINING"
echo "=========================================="

source $WARMUP_CONFIG

WARMUP_SAVE_DIR="$SCRATCH/Sparse-Attention-Zoo/checkpoints/run-${SLURM_JOB_ID}-warmup"
WARMUP_RUN_NAME="${WANDB_RUN_NAME}-${SLURM_JOB_ID}"

MAX_TRAIN_SAMPLES=$(($MAX_TRAIN_TOKENS / $MAX_SEQ_LENGTH))

echo "Warmup: Training on samples [${DATASET_OFFSET}:$((DATASET_OFFSET + MAX_TRAIN_SAMPLES))]"

WARMUP_ARGS=" \
    --model_name $MODEL_NAME \
    --index_top_k $INDEX_TOP_K \
    --index_num_heads $INDEX_NUM_HEADS \
    --index_head_dim $INDEX_HEAD_DIM \
    --rope_head_dim $ROPE_HEAD_DIM \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --dataset_name $DATASET_NAME \
    --dataset_config $DATASET_CONFIG \
    --dataset_split $DATASET_SPLIT \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --dataset_offset $DATASET_OFFSET \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WARMUP_RUN_NAME \
    --save_dir $WARMUP_SAVE_DIR \
    --save_every $SAVE_EVERY \
    --log_every $LOG_EVERY \
    --track_log_every $TRACK_LOG_EVERY \
    --weight_decay $WEIGHT_DECAY \
    --gradient_clipping $GRADIENT_CLIPPING \
    --warmup_stage \
    --track_token_selection
    "

srun --mpi=pmix --environment=pytorch2506 -u bash -lc '
set -x
source .venv/bin/activate
'"$LAUNCHER"' --machine_rank $SLURM_NODEID '"$SCRIPT $WARMUP_ARGS"'
'

WARMUP_EXIT=$?
if [ $WARMUP_EXIT -ne 0 ]; then
    echo "❌ Warmup stage failed with exit code $WARMUP_EXIT"
    exit $WARMUP_EXIT
fi

echo "✅ Stage 1 (Warmup) completed!"
WARMUP_CHECKPOINT="${WARMUP_SAVE_DIR}/final_model"

# ========================================
# STAGE 2: SPARSE TRAINING
# ========================================
echo "=========================================="
echo "STAGE 2: SPARSE TRAINING"
echo "=========================================="

source $SPARSE_CONFIG

SPARSE_SAVE_DIR="$SCRATCH/Sparse-Attention-Zoo/checkpoints/run-${SLURM_JOB_ID}-sparse"
SPARSE_RUN_NAME="${WANDB_RUN_NAME}-${SLURM_JOB_ID}"

# Use warmup's MAX_TRAIN_SAMPLES as offset
source $WARMUP_CONFIG
MAX_TRAIN_SAMPLES=$(($MAX_TRAIN_TOKENS / $MAX_SEQ_LENGTH))
WARMUP_SAMPLES=$MAX_TRAIN_SAMPLES

# Reload sparse config
source $SPARSE_CONFIG
export MAX_TRAIN_SAMPLES=$(($MAX_TRAIN_TOKENS / $MAX_SEQ_LENGTH))

echo "Sparse: Training on samples [${WARMUP_SAMPLES}:$((WARMUP_SAMPLES + MAX_TRAIN_SAMPLES))]"
echo "Loading checkpoint from: $WARMUP_CHECKPOINT"

SPARSE_ARGS=" \
    --model_name $MODEL_NAME \
    --index_top_k $INDEX_TOP_K \
    --index_num_heads $INDEX_NUM_HEADS \
    --index_head_dim $INDEX_HEAD_DIM \
    --rope_head_dim $ROPE_HEAD_DIM \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --dataset_name $DATASET_NAME \
    --dataset_config $DATASET_CONFIG \
    --dataset_split $DATASET_SPLIT \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --dataset_offset $WARMUP_SAMPLES \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $SPARSE_RUN_NAME \
    --save_dir $SPARSE_SAVE_DIR \
    --save_every $SAVE_EVERY \
    --log_every $LOG_EVERY \
    --track_log_every $TRACK_LOG_EVERY \
    --weight_decay $WEIGHT_DECAY \
    --gradient_clipping $GRADIENT_CLIPPING \
    --load_from_checkpoint $WARMUP_CHECKPOINT \
    --track_token_selection
    "

srun --mpi=pmix --environment=pytorch2506 -u bash -lc '
set -x
source .venv/bin/activate
'"$LAUNCHER"' --machine_rank $SLURM_NODEID '"$SCRIPT $SPARSE_ARGS"'
'

SPARSE_EXIT=$?
if [ $SPARSE_EXIT -ne 0 ]; then
    echo "❌ Sparse stage failed with exit code $SPARSE_EXIT"
    exit $SPARSE_EXIT
fi

echo "✅ Stage 2 (Sparse) completed!"
echo "=========================================="
echo "🎉 Two-stage training completed successfully!"
echo "Warmup checkpoint: $WARMUP_CHECKPOINT"
echo "Final checkpoint: ${SPARSE_SAVE_DIR}/final_model"
echo "=========================================="
