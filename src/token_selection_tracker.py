import torch
from typing import List
import matplotlib.pyplot as plt
import os
import wandb
import json


class TokenSelectionTracker:
    """
    Tracks which tokens are selected by the indexer at each layer for each training step.

    Stores the top-k indices directly for each layer at each step.
    """

    def __init__(self, top_k: int, layers: list[int], save_dir: str):
        self.top_k = top_k
        self.save_dir = save_dir
        self.layers = layers

        self.selection_heatmaps = []  # list[matplotlib.figure.Figure]
        self.selection_topks = []     # list[torch.Tensor]
        self._meta = []               # track (layer_idx, step) aligned with lists

        os.makedirs(self.save_dir, exist_ok=True)

    def record_selections(
        self,
        step: int,
        indexer_scores: List[torch.Tensor],
        wandb_run=None,
    ):
        for layer_idx, scores in enumerate(indexer_scores):
            if layer_idx not in self.layers or scores is None:
                continue

            batch_size, seq_len, _ = scores.shape
            scores_masked = scores.clone()

            # Apply causal mask
            causal_mask = torch.tril(torch.ones_like(scores_masked))
            scores_masked = scores_masked.masked_fill_(causal_mask == 0, float("-inf"))

            # Get top-k indices: (batch_size, seq_len, top_k)
            k = min(self.top_k, seq_len)
            _, top_k_indices = torch.topk(scores_masked, k=k, dim=-1, sorted=False)

            # Create binary selection mask
            selection_mask = torch.zeros_like(scores_masked)
            selection_mask.scatter_(-1, top_k_indices, 1.0)
            selection_mask = selection_mask.masked_fill_(causal_mask == 0, 0)

            # Average across batch to get selection frequency
            selection_freq = selection_mask.mean(dim=0).cpu()

            # Create heatmap
            fig = self._create_heatmap(selection_freq, layer_idx, step)

            # Keep in-memory (to be saved in save())
            self.selection_heatmaps.append(fig)
            self.selection_topks.append(top_k_indices.detach().cpu())
            self._meta.append({"layer": int(layer_idx), "step": int(step), "k": int(k), "seq_len": int(seq_len)})

            # Log to wandb if provided
            if wandb_run is not None:
                wandb_run.log(
                    {f"token_selection/hidden_layer_{layer_idx}": wandb.Image(fig)},
                    step=step,
                )

    def save(
        self,
        filename: str | None = None,
    ):
        """
        Save all collected tensors as lists in a single file (no images).
        - filename: path to the combined file; defaults to {save_dir}/run.pt
        """
        os.makedirs(self.save_dir, exist_ok=True)
        if filename is None:
            filename = os.path.join(self.save_dir, "run.pt")

        # Build payload: list of tensors + aligned metadata
        payload = {
            "meta": getattr(self, "_meta", []),           # list[dict]
            "topk": [t for t in self.selection_topks],    # list[Tensor]
            # If you also save selection_freq later, add another list here
            # "selection_freq": [freq.cpu() for freq in self.selection_freqs],
        }

        torch.save(payload, filename)

        self.reset()

    def reset(self):
        """
        Clear in-memory buffers.
        """
        self.selection_heatmaps = []
        self.selection_topks = []
        self._meta = []

    def _create_heatmap(self, selection_freq: torch.Tensor, layer_idx: int, step: int):
        """
        Create a heatmap figure showing token selection frequency.

        Args:
            selection_freq: Tensor (seq_len, seq_len) with selection frequencies [0, 1]
            layer_idx: Layer index
            step: Training step

        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(selection_freq.numpy(), aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Selection Frequency')
        plt.xlabel('Token Position')
        plt.ylabel('Query Position')
        plt.title(f'Token Selection Heatmap - Layer {layer_idx} (Step {step})')
        plt.tight_layout()
        return fig