import torch
from transformers import AutoModelForCausalLM
import time

from .dsa_llama_model import DSALlamaForCausalLM, DSALlamaConfig

def load_from_checkpoint(model_path: str):
    model = DSALlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    model.config._attn_implementation = "eager"

    return model


def create_dsa_llama_model_from_scratch(
    model_path: str,
    index_top_k: int,
    index_num_heads: int,
    rope_head_dim: int,
    index_head_dim: int):
    # Use official LLaMA 3.2 1B config (architecture only)
    config = DSALlamaConfig.from_pretrained(
        model_path,         
        index_top_k=index_top_k,
        index_num_heads=index_num_heads,
        rope_head_dim=rope_head_dim,
        index_head_dim=index_head_dim,
        )
    
    config._attn_implementation="eager"
    
    print("Model config:", config)

    # Build model from config (NO WEIGHTS LOADED)
    model = DSALlamaForCausalLM(config).to(torch.bfloat16)

    return model


def create_dsa_llama_model_pretrained(
    model_path: str,
    index_top_k: int,
    index_num_heads: int,
    rope_head_dim: int,
    index_head_dim: int):
    # Load config with your custom parameters
    config = DSALlamaConfig.from_pretrained(
        model_path,         
        index_top_k=index_top_k,
        index_num_heads=index_num_heads,
        rope_head_dim=rope_head_dim,
        index_head_dim=index_head_dim
    )
    config._attn_implementation="eager"

    # Create model with config
    model = DSALlamaForCausalLM(config).to(torch.bfloat16)
    
    # Load pretrained weights from original Llama
    from transformers import LlamaForCausalLM
    pretrained_model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    
    # Copy matching weights (embeddings, decoder layers, lm_head)
    model.model.embed_tokens.load_state_dict(pretrained_model.model.embed_tokens.state_dict())
    model.model.norm.load_state_dict(pretrained_model.model.norm.state_dict())
    model.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())
    
    # Load weights for each layer (except indexer which is new)
    for layer_idx in range(config.num_hidden_layers):
        # Load everything except self_attn (which has indexer)
        model.model.layers[layer_idx].input_layernorm.load_state_dict(
            pretrained_model.model.layers[layer_idx].input_layernorm.state_dict()
        )
        model.model.layers[layer_idx].post_attention_layernorm.load_state_dict(
            pretrained_model.model.layers[layer_idx].post_attention_layernorm.state_dict()
        )
        model.model.layers[layer_idx].mlp.load_state_dict(
            pretrained_model.model.layers[layer_idx].mlp.state_dict()
        )
        
        # Load attention weights (q, k, v, o projections)
        model.model.layers[layer_idx].self_attn.q_proj.load_state_dict(
            pretrained_model.model.layers[layer_idx].self_attn.q_proj.state_dict()
        )
        model.model.layers[layer_idx].self_attn.k_proj.load_state_dict(
            pretrained_model.model.layers[layer_idx].self_attn.k_proj.state_dict()
        )
        model.model.layers[layer_idx].self_attn.v_proj.load_state_dict(
            pretrained_model.model.layers[layer_idx].self_attn.v_proj.state_dict()
        )
        model.model.layers[layer_idx].self_attn.o_proj.load_state_dict(
            pretrained_model.model.layers[layer_idx].self_attn.o_proj.state_dict()
        )
        # Note: indexer weights remain randomly initialized
    
    del pretrained_model  # Free memory
    
    return model


# From accelerate/examples
class PerformanceTracker:
    """Track training performance metrics."""

    def __init__(self, warmup_steps: int = 10):
        self.warmup_steps = warmup_steps
        self.reset()

    def reset(self):
        """Reset all tracking variables."""
        self.start_time = None
        self.num_tokens = 0
        self.is_in_warmup = True
        self.step_count = 0

    def step(self, batch_tokens: int, model_flops_per_token: float | None = None) -> dict:
        """
        Update performance tracking with a new step.

        Args:
            batch_tokens (int): Number of tokens in current batch

        Returns:
            dict: Performance metrics if past warmup, empty dict otherwise
        """
        self.step_count += 1

        if self.step_count == self.warmup_steps:
            self.start_time = time.perf_counter()
            self.num_tokens = 0
            self.is_in_warmup = False
            return {"warmup_completed": True}

        if not self.is_in_warmup and self.start_time is not None:
            dct = {}
            self.num_tokens += batch_tokens
            total_time = time.perf_counter() - self.start_time
            steps_from_warmup = self.step_count - self.warmup_steps

            if total_time > 0 and steps_from_warmup > 0:
                dct = {
                    "tokens_per_second": self.num_tokens / total_time,
                    "steps_per_second": steps_from_warmup / total_time,
                    "total_tokens": self.num_tokens,
                    "total_time": total_time,
                }

            if model_flops_per_token is not None:
                flops = model_flops_per_token * self.num_tokens
                dct["tflops_per_device"] = flops / (total_time * 1e12)

            return dct

        return {}
    

def get_model_flops_per_token(model: AutoModelForCausalLM, seq_len: int) -> float:
    """
    Get the number of flops per token for the model.

    Args:
        model (AutoModelForCausalLM): Model to get the flops for
        seq_len (int): Sequence length
    """

    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    # MLP: 3 matmuls
    mlp_flops = 18 * cfg.hidden_size * cfg.intermediate_size

    # Indexer: 3 projections + attention computation
    # q_proj: hidden_size -> num_heads * head_dim (6 FLOPs per element)
    # k_proj: hidden_size -> head_dim (6 FLOPs per element)
    # w_proj: hidden_size -> num_heads (6 FLOPs per element)
    # indexer_proj_flops = 6 * cfg.hidden_size * (index_num_heads * index_head_dim + index_head_dim + index_num_heads)
    # # Attention: q @ k^T scales with sequence length (12 * num_heads * head_dim * seq_len)
    # indexer_attn_flops = 12 * index_num_heads * index_head_dim * seq_len
    # indexer_flops = indexer_proj_flops + indexer_attn_flops

    # Attn (w/o dotproduct)
    attn_flops = 12 * head_dim * (cfg.num_attention_heads + cfg.num_key_value_heads)

    # attn (dotproduct) - this scales quadratically with sequence length
    attn_dotproduct_flops = 12 * cfg.num_attention_heads * head_dim * seq_len

    # we also ignore embeddings and layernorms, indexer, etc
    return (mlp_flops + attn_flops + attn_dotproduct_flops) * cfg.num_hidden_layers


def get_model_size_breakdown(model: torch.nn.Module) -> str:
    """
    Generate a detailed model size breakdown string.

    Args:
        model: The model to analyze

    Returns:
        str: Formatted string with model size breakdown
    """
    lines = []

    def format_params(n):
        """Format parameter count in millions and billions."""
        if n >= 1e9:
            return f"{n / 1e9:.3f}B"
        else:
            return f"{n / 1e6:.2f}M"

    # Get parameter counts per module
    module_params = {}
    for name, param in model.named_parameters():
        parts = name.split('.')
        # Group by layer and component
        if len(parts) >= 3 and parts[1] == 'layers':
            layer_idx = parts[2]
            component = '.'.join(parts[3:-1])  # e.g., 'self_attn.q_proj'
            key = f"layers.{layer_idx}.{component}"
        else:
            key = '.'.join(parts[:-1])  # Everything except the final weight/bias

        if key not in module_params:
            module_params[key] = 0
        module_params[key] += param.numel()

    total_params = sum(module_params.values())

    lines.append("="*80)
    lines.append(f"Model Size: {format_params(total_params)} ({total_params:,} params)")
    lines.append("="*80)

    # Group by layer
    from collections import defaultdict
    layers_dict = defaultdict(dict)
    non_layer_params = {}

    for key, count in sorted(module_params.items()):
        if key.startswith('model.layers.'):
            parts = key.split('.')
            layer_idx = int(parts[2])
            component = '.'.join(parts[3:])
            layers_dict[layer_idx][component] = count
        else:
            non_layer_params[key] = count

    # Print non-layer params
    if non_layer_params:
        lines.append("\nNon-layer parameters:")
        for key, count in sorted(non_layer_params.items()):
            lines.append(f"  {key}: {format_params(count)}")

    # Print per-layer breakdown
    if layers_dict:
        lines.append(f"\nPer-layer breakdown ({len(layers_dict)} layers):")
        lines.append("-"*80)

        for layer_idx in sorted(layers_dict.keys()):
            components = layers_dict[layer_idx]
            layer_total = sum(components.values())
            lines.append(f"\n[Layer {layer_idx}] Total: {format_params(layer_total)}")

            for component, count in sorted(components.items()):
                lines.append(f"  {component}: {format_params(count)}")

    lines.append("\n" + "="*80)

    return '\n'.join(lines)
