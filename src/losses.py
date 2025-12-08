import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, Optional

def compute_indexer_kl_loss(attention_scores, indexer_scores, top_k=None):
    """
    Compute KL divergence loss to align indexer with main attention.
    This is the L_I loss from the paper:
    - Dense warmup: L_I = Σ_t KL(p_{t,:} || Softmax(I_{t,:}))
    - Sparse training: L_I = Σ_t KL(p_{t,S_t} || Softmax(I_{t,S_t}))

    Args:
        attention_scores: tensor of shape (batch, num_heads, seq_len, seq_len)
        indexer_scores: tensor of shape (batch, seq_len, seq_len) (this are the logits, and they already causal masked)
        top_k: if provided, only compute loss on top-k selected tokens (sparse training)
    """
    # Sum attention across all heads: (batch, num_heads, seq_len, seq_len) -> (batch, seq_len, seq_len)
    target_dist = attention_scores.sum(dim=1)

    # L1 normalize along sequence dimension (creates a valid probability distribution)
    target_dist = F.normalize(target_dist, p=1, dim=-1) # scores should already by positive (so abs is not necessary in reality)

    if top_k is not None:
        # Recreate the sparse mask based on top-k indexer scores
        total_seq_len = indexer_scores.shape[-1]
        _, top_k_indices = torch.topk(indexer_scores, k=min(top_k, total_seq_len), dim=-1, sorted=False)
        sparse_mask = torch.full_like(indexer_scores, False, dtype=torch.bool)
        sparse_mask = sparse_mask.scatter_(-1, top_k_indices, True)  # True for top-k selected positions

        # Apply sparse mask to indexer scores and compute log_softmax
        index_scores_masked = indexer_scores.masked_fill(~sparse_mask, torch.finfo(indexer_scores.dtype).min)
        indexer_distribution = F.log_softmax(index_scores_masked, dim=-1, dtype=torch.float32).to(indexer_scores.dtype) # Changed the dtype to float32 for numerical stability of softmax

        # Mask and renormalize target distribution
        # NOTE: in the DSA calculation we already mask every head the same way, so summing across heads should not introduce non-zero values in the masked positions (But I don't know if I should leave it just in case)
        # target_dist_masked = target_dist.masked_fill(~sparse_mask, 0.0)
        # target_dist_masked = F.normalize(target_dist_masked, p=1, dim=-1)

        p_target_distribution = target_dist
    else:
        # Dense warm-up: full sequence
        indexer_distribution = F.log_softmax(indexer_scores, dim=-1, dtype=torch.float32).to(indexer_scores.dtype)
        p_target_distribution = target_dist

    # KL divergence: KL(target || indexer) (p || q)
    # We do a sum over the tokens and then we do a mean over the batch dimension
    kl_loss = F.kl_div(
            indexer_distribution,
            p_target_distribution,
            reduction='batchmean',
            log_target=False, # Target distribution is not in log space
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