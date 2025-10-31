import torch
from typing import Optional, List
from collections import defaultdict


class TokenSelectionTracker:
    """
    Tracks which tokens are selected by the indexer at each layer for each training step.

    Stores the top-k indices directly for each layer at each step.
    """

    def __init__(self, max_seq_len: int, top_k: int, num_layers: int):
        """
        Initialize the token selection tracker.

        Args:
            max_seq_len: Maximum sequence length to track
            top_k: Number of tokens selected by the indexer
            num_layers: Number of decoder layers in the model
        """
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        self.num_layers = num_layers

        # Store top-k indices per layer
        # selections[layer_idx] = [(step, top_k_indices), ...]
        # top_k_indices shape: (batch_size, seq_len, top_k)
        self.selections = [[] for _ in range(num_layers)]

    def record_selections(
        self,
        step: int,
        indexer_scores: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Record which tokens were selected at this training step.

        Args:
            step: Current training step
            indexer_scores: List of tensors (one per layer), shape (batch_size, seq_len, seq_len)
            attention_mask: Optional mask (4D or 2D)
        """
        for layer_idx, scores in enumerate(indexer_scores):
            if scores is None:
                continue

            batch_size, seq_len, _ = scores.shape
            scores_masked = scores.clone()

            # Get top-k indices: (batch_size, seq_len, top_k)
            _, top_k_indices = torch.topk(
                scores_masked[:, -1, :],
                k=min(self.top_k, seq_len),
                dim=-1,
                sorted=False
            ) # (batch_size, top_k)

            self.selections[layer_idx].append((step, top_k_indices[0].cpu())) # top_K indices

    def get_selection_indices(self, step: int, layer_idx: int) -> Optional[torch.Tensor]:
        """Get top-k indices for a specific step and layer."""
        for recorded_step, indices in self.selections[layer_idx]:
            if recorded_step == step:
                return indices
        return None

    def save(self, filepath: str):
        """
        Convert each layer's (step, [token_indices]) records into
        a sparse COO tensor of shape (max_step+1, seq_len)
        and torch.save the list to `filepath`.
        """
        sparse_by_layer = []
        # Find the maximum step across all layers so we can fix the first dim
        all_steps = [
            step
            for layer_recs in self.selections
            for (step, _) in layer_recs
        ]
        max_step = max(all_steps) if all_steps else -1

        for layer_idx in range(self.num_layers):
            recs = self.selections[layer_idx]
            if not recs:
                # no selections for this layer
                sparse_by_layer.append(
                    torch.sparse_coo_tensor(
                        torch.empty((2, 0), dtype=torch.long),
                        torch.tensor([], dtype=torch.float32),
                        size=(max_step + 1, self.max_seq_len),
                    )
                )
                continue

            rows = []
            cols = []
            for step, topk_idxs in recs:
                # topk_idxs is a 1D CPU tensor of length ≤ top_k
                for token_idx in topk_idxs.tolist():
                    rows.append(step)
                    cols.append(token_idx)

            # Build the 2×nnz indices matrix
            indices = torch.tensor([rows, cols], dtype=torch.long)
            # Values can all be 1.0 to mark “selected”
            values = torch.ones(indices.shape[1], dtype=torch.float32)
            size = (max_step + 1, self.max_seq_len)

            sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
            sparse_by_layer.append(sparse_tensor)

        # Now save the list of sparse tensors (and any metadata you want)
        torch.save(
            {
                "sparse_selections": sparse_by_layer,
                "num_layers": self.num_layers,
                "seq_len": self.max_seq_len,
                "top_k": self.top_k,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> 'TokenSelectionTracker':
        """
        Load tracker state from disk (saved via `save`) and rebuild
        `tracker.selections[layer] = [(step, topk_indices_tensor), …]`.
        """
        state = torch.load(filepath)
        num_layers = state["num_layers"]
        seq_len = state["seq_len"]
        top_k = state["top_k"]
        sparse_by_layer = state["sparse_selections"]

        # Reconstruct the tracker
        tracker = cls(
            max_seq_len=seq_len,
            top_k=top_k,
            num_layers=num_layers
        )

        # For each layer, group the sparse indices by step
        for layer_idx, sp in enumerate(sparse_by_layer):
            # Ensure coalesced so indices() are unique
            sp = sp.coalesce()
            rows, cols = sp.indices()    # both are 1D LongTensors of size nnz

            # Group token cols by their step (row)
            step_to_tokens = defaultdict(list)
            for step, token in zip(rows.tolist(), cols.tolist()):
                step_to_tokens[step].append(token)

            # Rebuild the recorded list in ascending step order
            for step in sorted(step_to_tokens):
                tokens = step_to_tokens[step]
                # Turn into a LongTensor; if you want sorted or padded, do it here
                idx_tensor = torch.tensor(tokens, dtype=torch.long)
                tracker.selections[layer_idx].append((step, idx_tensor))

        return tracker

    def reset(self):
        """Reset all tracking data."""
        self.selections = [[] for _ in range(self.num_layers)]

    def num_steps(self) -> int:
        """Return the number of unique steps tracked."""
        all_steps = set()
        for layer_selections in self.selections:
            for step, _ in layer_selections:
                all_steps.add(step)
        return len(all_steps)

    def __repr__(self) -> str:
        return (
            f"TokenSelectionTracker(max_seq_len={self.max_seq_len}, "
            f"top_k={self.top_k}, num_layers={self.num_layers}, "
            f"num_steps={self.num_steps()})"
        )
