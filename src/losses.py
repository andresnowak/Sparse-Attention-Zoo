import torch
import torch.nn.functional as F
from typing import Tuple

def compute_indexer_kl_loss(attention_scores, indexer_scores, top_k=None):
    """
    Args:
        attention_scores: tensor of shape (batch, num_heads, seq_len, seq_len)
        indexer_scores: tensor of shape (batch, seq_len, seq_len)
        top_k: if provided, only compute loss on top-k selected tokens (sparse training)
    """
    total_loss = 0
    
    # Sum attention across all heads: (batch, num_heads, seq_len, seq_len) -> (batch, seq_len, seq_len)
    p_target = attention_scores.sum(dim=1)

    # L1 normalize along sequence dimension
    p_target = F.normalize(p_target, p=1, dim=-1)
    
    if top_k is not None:
        # Sparse training: only consider selected tokens
        _, top_k_indices = torch.topk(indexer_scores, k=top_k, dim=-1)
        # Gather only selected positions
        p_target_selected = torch.gather(p_target, -1, top_k_indices)
        index_scores_selected = torch.gather(indexer_scores, -1, top_k_indices)
        
        kl_loss = F.kl_div(
            F.log_softmax(index_scores_selected, dim=-1),
            p_target_selected,
            reduction='batchmean'
        )
    else:
        # Dense warm-up: full sequence
        kl_loss = F.kl_div(
            F.log_softmax(indexer_scores, dim=-1),
            p_target,
            reduction='batchmean'
        )
    
    return kl_loss

def cross_entropy_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    ce_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )

    # probs = torch.nn.functional.softmax(shift_logits, dim=-1)
    # log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    return ce_loss

def warmup_stage_loss(logits: torch.Tensor, labels: torch.Tensor, kl_loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ce_loss = cross_entropy_loss(logits, labels)

    return ce_loss, kl_loss
