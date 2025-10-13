# model.py
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils.generic import TransformersKwargs, can_return_tuple
from transformers.processing_utils import Unpack
from transformers import LlamaConfig, LlamaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention, apply_rotary_pos_emb, eager_attention_forward, LlamaDecoderLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Union
from collections.abc import Callable


class DSALlamaConfig(LlamaConfig):
    model_type = "llama"  # Keep same model type
    
    def __init__(
        self,
        index_top_k=2048,
        num_index_heads=1,
        rope_head_dim=32,
        index_hidden_size=64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.index_top_k = index_top_k
        self.num_index_heads = num_index_heads
        self.rope_head_dim = rope_head_dim
        self.index_hidden_size = index_hidden_size


class Indexer(nn.Module):
    def __init__(self, config: DSALlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.indexer_cache_idx = self.layer_idx + config.num_hidden_layers

        self.config = config
    
        self.hidden_size = config.hidden_size
        self.index_hidden_size = config.index_hidden_size
        self.num_heads = config.num_index_heads
        self.head_dim = self.index_hidden_size // self.num_heads
        

        self.rope_head_dim = config.rope_head_dim 
        if self.rope_head_dim > self.head_dim:
            raise ValueError(f"rope_head_dim ({self.rope_head_dim}) cannot be larger than the indexer head_dim ({self.head_dim})")

        self.k_norm = nn.LayerNorm(self.head_dim)

        self.q_index_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_index_proj = nn.Linear(self.hidden_size, self.head_dim, bias=config.attention_bias)
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

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, 1, self.head_dim)

        cos_partial = cos[..., :self.rope_head_dim]
        sin_partial = sin[..., :self.rope_head_dim]
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_partial, sin_partial, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, _ = past_key_values.update(k, torch.empty(), self.indexer_cache_idx, cache_kwargs)

        # 2. Reshape for multi-head processing
        # Reshape q to separate the heads
        batch, seq_len, _, _ = q.shape
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim) # (batch, seq_len, num_heads, head_dim)
        q = q.permute(0, 2, 1, 3) # (batch, num_heads, seq_len, head_dim)

        score = F.relu(q @ k.transpose(-2, -1)) # (batch, num_heads, seq_len, seq_len)

        indexer_score = (w.transpose(-2, -1).unsqueeze(1) * score).sum(dim=1) # (batch, seq_len, seq_len)

        return indexer_score


class LlamaDSA(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper modified to use partial rope"""
    def __init__(self, config: DSALlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.index_top_k = config.index_top_k
        self.rope_head_dim = config.rope_head_dim 
        self.indexer = Indexer(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, seq_len, _ = hidden_states.shape

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (batch_size, num_heads, seq_len, head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        # Indexer

        cos_partial = cos[..., self.rope_head_dim::]
        sin_partial = sin[..., self.rope_head_dim::]

        q_nope, q_pe = torch.split(query_states, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_nope, k_pe = torch.split(key_states, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_partial, sin_partial)
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        # Indexer 
        index_scores = self.indexer(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if attention_mask is not None:
            causal_mask = attention_mask[:, 0, :, :].squeeze(1)
            index_scores = index_scores + causal_mask

        with torch.no_grad():
            _, top_k_indices = torch.topk(index_scores, k=min(self.index_top_k, seq_len), dim=-1)

        sparse_mask = torch.full_like(index_scores, -float("inf"))
        sparse_mask = sparse_mask.scatter_(-1, top_k_indices, 0.0)

        if attention_mask != None:  
            attention_mask = attention_mask + sparse_mask.unsqueeze(1)
        else:
            attention_mask = sparse_mask.unsqueeze(1)

        # ---

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    

class DSALlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: DSALlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.index_top_k = config.index_top_k
        
        self.self_attn = LlamaDSA(config=config, layer_idx=layer_idx)
        self.indexer = Indexer(config=config, layer_idx=layer_idx)
        

class DSALlamaModel(LlamaModel):
    def __init__(self, config: DSALlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [DSALlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = DSALlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )