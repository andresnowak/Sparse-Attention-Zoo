import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import argparse
from dotenv import load_dotenv
from transformers import PreTrainedTokenizer
from datasets import load_dataset


def get_dataloader(
    accelerator: Accelerator,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    text_column: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    max_samples: int | None = None,
    offset: int = 0
) -> DataLoader:
    """
    Creates DataLoader for causal LM with dynamic batching and hardware-aware padding.
    """
    # Load dataset with offset and max_samples
    if max_samples:
        split_str = f"{dataset_split}[{offset}:{offset + max_samples}]"
    elif offset > 0:
        split_str = f"{dataset_split}[{offset}:]"
    else:
        split_str = dataset_split

    with accelerator.main_process_first():
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split_str,
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
        tokenized = dataset.map(
            tokenize,
            batched=True,
            num_proc=32,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {dataset_split}",
        )

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
        num_workers=2,
        prefetch_factor=2,  # Each worker prefetches 2 batches
        persistent_workers=True  # Keep workers alive between epochs
    )
