import torch

from .dsa_llama_model import DSALlamaForCausalLM, DSALlamaConfig

def create_dsa_llama_model_from_scratch(
    model_path: str,
    index_top_k: int,
    num_index_heads: int,
    rope_head_dim: int,
    index_hidden_size: int):
    # Use official LLaMA 3.2 1B config (architecture only)
    config = DSALlamaConfig.from_pretrained(
        model_path,         
        index_top_k=index_top_k,
        num_index_heads=num_index_heads,
        rope_head_dim=rope_head_dim,
        index_hidden_size=index_hidden_size
        )
    
    # Optional: inspect it
    print("Model config:", config)

    # Build model from config (NO WEIGHTS LOADED)
    model = DSALlamaForCausalLM(config)

    return model


def create_dsa_llama_model_pretrained(
    model_path: str,
    index_top_k: int,
    num_index_heads: int,
    rope_head_dim: int,
    index_hidden_size: int):
    # Load config with your custom parameters
    config = DSALlamaConfig.from_pretrained(
        model_path,         
        index_top_k=index_top_k,
        num_index_heads=num_index_heads,
        rope_head_dim=rope_head_dim,
        index_hidden_size=index_hidden_size
    )

    # Create model with config
    model = DSALlamaForCausalLM(config)
    
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