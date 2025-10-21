# Sparse Attention Zoo

## Currently Implemented

### DeepSeek-V3.2 Dynamic Sparse Attention (DSA)

Implementation of the Dynamic Sparse Attention mechanism from [DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf).

**Key Features:**
- **Lightning Indexer**: Lightweight attention indexer that selects relevant tokens
- **Two-Stage Training**:
  - **Dense Warm-up Stage**: Train only the indexer with frozen main model using dense attention
  - **Sparse Training Stage**: Train both indexer and main model with sparse attention pattern
- **Efficient Token Selection**: Uses top-k selection from indexer scores
- **KL Divergence Alignment**: Aligns indexer distribution with main attention distribution

**Architecture:**
- Separate indexer network per attention layer
- Partial RoPE application for position encoding
- Detached indexer input from main computational graph
- Multi-head indexer with configurable heads and dimensions

## Installation

```bash
uv sync
```

and NGC 25.06 pytorch container

## Usage

### Key Arguments

**Model Configuration:**
- `--model_name`: Base LLaMA model to use
- `--index_top_k`: Number of tokens to select per query (default: 2048)
- `--index_num_heads`: Number of indexer heads (default: 16)
- `--rope_head_dim`: Dimension for RoPE in indexer (default: 32)
- `--index_head_dim`: Head dimension for indexer (default: 64)

**Training Configuration:**
- `--warmup_stage`: Enable dense warm-up stage (freeze main model)
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate
- `--num_epochs`: Number of training epochs
- `--gradient_accumulation_steps`: Gradient accumulation steps

**Data Configuration:**
- `--dataset_name`: HuggingFace dataset name
- `--dataset_config`: Dataset configuration
- `--max_train_samples`: Maximum number of training samples
- `--dataset_offset`: Offset for dataset samples

**Logging:**
- `--wandb_project`: W&B project name
- `--wandb_run_name`: W&B run name
- `--log_every`: Logging frequency (steps)

## Architecture Details

### Indexer Network
- Query projection: `hidden_size, num_heads * head_dim`
- Key projection: `hidden_size, head_dim`
- Weight projection: `hidden_size, num_heads`
- Partial RoPE on first `rope_head_dim` dimensions
- LayerNorm on keys

### Loss Functions
- **Warm-up Stage**: KL divergence between indexer and aggregated attention scores
- **Sparse Stage**:
  - Main model: Cross-entropy language modeling loss
  - Indexer: KL divergence on selected top-k tokens only

## References

- [DeepSeek-V3.2 Paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)
- DeepSeek-V3.2: "We first use a short warm-up stage to initialize the lightning indexer. In this stage, we keep dense attention and freeze all model parameters except for the lightning indexer."