# indexer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


class Indexer(nn.Module):
    # Lightning Indexer for DeepSeek Sparse Attention
    
    # Computes index scores I_{t,s} = Σ w_{t,j} · ReLU(q_{t,j} · k_s)
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.indexer_cache_idx = self.layer_idx + config.num_hidden_layers

        self.config = config

        self.hidden_size = config.hidden_size # Model hidden size
        self.num_heads = config.index_num_heads
        self.head_dim = config.index_head_dim

        # Scaling factors for the "attention" logits
        self.softmax_scale = self.head_dim ** -0.5 # scale for the q · k in the indexer
        self.n_heads_scale = self.num_heads ** -0.5

        self.rope_head_dim = config.rope_head_dim
        if self.rope_head_dim > self.head_dim:
            raise ValueError(f"rope_head_dim ({self.rope_head_dim}) cannot be larger than the indexer head_dim ({self.head_dim})")

        self.k_norm = nn.LayerNorm(self.head_dim)

        # Query indexer projection: h_t -> {q_{t,j}^I}
        self.q_index_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # Key indexer projection: h_s -> k_s^I
        self.k_index_proj = nn.Linear(self.hidden_size, self.head_dim, bias=config.attention_bias)
        # Weights indexer: w_{t,j}^I for each head
        self.weights_index_proj = nn.Linear(self.hidden_size, self.num_heads, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # hidden_states has shape (batch_size, seq_len, hidden_size)

        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # 1. Calculate q, k, and w
        q: torch.Tensor = self.q_index_proj(hidden_states)  # Shape: (batch, seq_len, n_heads * head_dim)
        k: torch.Tensor = self.k_index_proj(hidden_states)  # Shape: (batch, seq_len, head_dim) 
        k = self.k_norm(k)
        w: torch.Tensor = self.weights_index_proj(hidden_states) # Shape: (batch, seq_len, n_heads)
        w = self.softmax_scale * self.n_heads_scale * w # Doing the scaling before doing the sum w_i * ReLU(q_i * k_i)
        # NOTE: The scaling is done because we are trying to replicate the distriubtion of dense attention scores, but also because here in the indexer the variance also grows by head_dim (if we have q_i and k_i with variance 1, then q_i * k_i has variance head_dim, and then summing over n_heads also increases variance by n_heads (even though RELU changes our variance and mean))

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, 1, self.head_dim)

        # Apply RoPE (partial or full based on config)
        if self.rope_head_dim < self.head_dim:
            # Partial RoPE: only apply to first rope_head_dim dimensions
            cos_partial = cos[..., :self.rope_head_dim]
            sin_partial = sin[..., :self.rope_head_dim]
            q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
            k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_partial, sin_partial, unsqueeze_dim=2)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)
        else: # rope_dim = head_dim
            # Full RoPE: apply to all dimensions
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # 2. Reshape for multi-head processing
        # Reshape q to separate the heads
        batch, seq_len, _, _ = q.shape
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim) # (batch, seq_len, num_heads, head_dim)
        q = q.permute(0, 2, 1, 3) # (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2) # (batch, 1, seq_len, head_dim)

        # we add the key after its transpose (butr we could instead not save the transpose and we can have the cache before this part)
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # NOTE: we don't have a weight cache because weight is per query, in the hadamard product weight is repeated for all keys
            k, _ = past_key_values.update(k, torch.empty(0, dtype=k.dtype, device=k.device), self.indexer_cache_idx, cache_kwargs)

        score = F.relu(q @ k.transpose(-2, -1)) # (batch, num_heads, seq_len, seq_len)

        # indexer_score = (w.transpose(-2, -1).unsqueeze(-1) * score).sum(dim=1) # (batch, seq_len, seq_len), the logits
        indexer_score = torch.einsum('bqh,bhqk->bqk', w, score)  # (batch, seq_len, seq_len), the logits
        # indexer_score = (w.unsqueeze(2) @ score.transpose(-3, -2)).squeeze(2) # we do first the multiplication of the weights for all heads by the keys in all heads, and then we sum over heads (the weighted sum over heads for each query)

        return indexer_score