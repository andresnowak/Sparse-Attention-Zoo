import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, Optional

def compute_indexer_kl_loss(attention_scores, indexer_scores, top_k=None):
    """
    Args:
        attention_scores: tensor of shape (batch, num_heads, seq_len, seq_len)
        indexer_scores: tensor of shape (batch, seq_len, seq_len)
        top_k: if provided, only compute loss on top-k selected tokens (sparse training)
    """
    # Compute KL divergence loss to align indexer with main attention
    # This is the L_I loss from the paper:
    # - Dense warmup: L_I = Σ_t KL(p_{t,:} || Softmax(I_{t,:}))
    # - Sparse training: L_I = Σ_t KL(p_{t,S_t} || Softmax(I_{t,S_t}))

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
            reduction='batchmean',
            log_target=False,
        )
    
    return kl_loss


# Huggingface
def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    # NOTE: Because we are doing DDP, by default the reduce of the gradients is done with a mean (so we don't want our loss reduction to be a sum or we need lr to be multiplied by world size of DP)
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction=reduction
    )
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(
        logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    )
    return loss