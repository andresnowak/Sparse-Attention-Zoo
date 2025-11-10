import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import argparse
from dotenv import load_dotenv
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from datatrove.utils.dataset import DatatroveFileDataset, DatatroveFolderDataset


def get_dataloader(
    accelerator: Accelerator,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    text_column: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int, # seq_length
    batch_size: int,
    shuffle: bool,
    max_samples: int | None = None,
    offset: int = 0,
    data_folder: str | None = None,
) -> DataLoader:
    """
    Creates DataLoader for causal LM with dynamic batching and hardware-aware padding.
    """

    # Load full dataset
    with accelerator.main_process_first():
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
        )

    def tokenize(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_overflowing_tokens=False,
            add_special_tokens=True
        )

    # Tokenize on main process first, others wait and use cache
    with accelerator.main_process_first():
        # Only use multiprocessing on main process to avoid resource contention
        num_proc = 64

        tokenized = dataset.map(
            tokenize,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {dataset_split}",
        )

        # Filter out examples shorter than max_length (we want all examples to have the correct size)
        pre_filter_count = len(tokenized)
        tokenized = tokenized.filter(
            lambda x: len(x["input_ids"]) >= max_length,
            num_proc=num_proc,
            desc=f"Filtering sequences < {max_length} tokens",
        )
        post_filter_count = len(tokenized)

        accelerator.print(f"Filtered: {pre_filter_count} -> {post_filter_count} samples ({pre_filter_count - post_filter_count} removed)")

        # Apply offset and max_samples after filtering
        if offset > 0 or max_samples:
            start_idx = offset
            end_idx = offset + max_samples if max_samples else post_filter_count

            assert end_idx <= post_filter_count, \
                f"Not enough samples: need index {end_idx} (offset={offset}, max_samples={max_samples}), but only {post_filter_count} available after filtering"

            tokenized = tokenized.select(range(start_idx, end_idx))
            accelerator.print(f"Selected samples [{start_idx}:{end_idx}]")

    def collate_batch(examples):
        # Dynamic padding: pad to longest in batch
        padded = tokenizer.pad(
            examples,
            padding="longest",
            max_length=None,
            pad_to_multiple_of=8 if accelerator.mixed_precision in ["fp16", "bf16"] else None,
            return_tensors="pt"
        )

        # Create labels (same as input_ids but mask padding)
        labels = padded["input_ids"].clone()
        labels[padded["attention_mask"] == 0] = -100  # Standard CLM masking
        padded["labels"] = labels
        
        return padded

    # Create DataLoader with prefetching for faster data loading
    # Note: num_workers creates parallel processes per GPU rank
    return DataLoader(
        tokenized,
        shuffle=shuffle,
        collate_fn=collate_batch,
        batch_size=batch_size,
        drop_last=dataset_split == "train",
        pin_memory=True,
        num_workers=1,
        prefetch_factor=2,  # Each worker prefetches 2 batches
        # persistent_workers=True  # Keep workers alive between epochs
    )
