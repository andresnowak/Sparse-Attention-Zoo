# model.py
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.utils.generic import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers import LlamaConfig, LlamaForCausalLM
import torch.nn as nn
import math
import torch
from typing import Optional

class CustomLLMConfig(PretrainedConfig):
    model_type = "custom_llm"
    def __init__(
        self,
        vocab_size=128256,          # ⬅️ CRITICAL: match Llama tokenizer
        hidden_size=2048,
        num_hidden_layers=16,       # ⬅️ From your JSON config
        num_attention_heads=32,     # ⬅️ From your JSON
        intermediate_size=8192,
        max_position_embeddings=131072,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

class CustomLLM(PreTrainedModel):
    config_class = CustomLLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                batch_first=True,
                activation="gelu"
            ) for _ in range(config.num_hidden_layers)
        ])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()  # Triggers weight initialization

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.config.num_hidden_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, x)  # Self-attention
        logits = self.lm_head(x)

        if labels is not None:
            return {"logits": logits, "labels": labels}
        return {"logits": logits}
    

# Custom sparse attention (example)
class MyCustomAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Custom logic here (e.g., sparse masking)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Implement custom attention logic (shortened)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Your sparse attention / custom RoPE / etc.

        out = self.o_proj(v)  # Simplified
        return out, None
    

class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.model.layers[5].self_attn = MyCustomAttention(config)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Call original forward (or override as needed)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )