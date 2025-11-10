"""
Dataset tokenization using Datatrove for Hugging Face datasets.
"""

import argparse
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import DocumentTokenizer
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter


def tokenize_hf_dataset(
    dataset_name: str,
    subset: str,
    output_path: str,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    min_tokens_per_sample: int = 2048,
    limit: int = -1,
):
    """
    Tokenize a Hugging Face dataset using Datatrove.

    Args:
        dataset_name: Name of the HF dataset (e.g., "wikimedia/wikipedia")
        subset: Dataset subset (e.g., "CC-MAIN-2024-10" or "sample/100BT")
        output_path: Where to save tokenized output
        tokenizer_name: HF tokenizer to use
        min_tokens_per_sample: Minimum tokens per document to keep
        limit: Limit number of documents to process (None for all)
    """

    # Define the pipeline
    pipeline=[
        # Read parquet files from HuggingFace dataset
        ParquetReader(f"hf://datasets/{dataset_name}/{subset}", limit=limit),
        # Filter documents based on pre-existing token_count from GPT-2 tokenizer
        LambdaFilter(lambda doc: (doc.metadata.get('token_count') or 0) >= min_tokens_per_sample),
        # Tokenize documents with specified tokenizer
        DocumentTokenizer(
            output_folder=output_path,
            tokenizer_name_or_path=tokenizer_name,
            eos_token="<|end_of_text|>",
        ),
        # # Save tokenized output as JSONL
        # JsonlWriter(output_path)
    ]

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a Hugging Face dataset using Datatrove")

    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the HF dataset (e.g., 'HuggingFaceFW/fineweb-edu')")
    parser.add_argument("--subset", type=str, required=True, help="Dataset subset (e.g., 'sample/100BT' or 'CC-MAIN-2024-10')")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save tokenized output")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.2-1B", help="HF tokenizer to use")
    parser.add_argument("--min_tokens_per_sample", type=int, default=2048, help="Minimum tokens per document to keep")
    parser.add_argument("--num_tasks", type=int, default=1, help="Number of tasks")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to use for parallel work")
    parser.add_argument("--partition", type=str, default="normal", help="Slurm partition to use")
    parser.add_argument("--time", type=str, default="24:00:00", help="Job time limit (format: HH:MM:SS)")
    parser.add_argument("--cpus_per_task", type=int, default=30, help="Number of CPUs per task")
    parser.add_argument("--mem_per_cpu_gb", type=int, default=4, help="Memory per CPU in GB")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of documents to process (None for all)")

    args = parser.parse_args()

    pipeline = tokenize_hf_dataset(
        dataset_name=args.dataset_name,
        subset=args.subset,
        output_path=args.output_path,
        tokenizer_name=args.tokenizer_name,
        min_tokens_per_sample=args.min_tokens_per_sample,
        limit=args.limit,
    )

    # Execute the pipeline
    executor = SlurmPipelineExecutor(
        pipeline=pipeline,
        tasks=args.num_tasks,
        partition=args.partition,
        logging_dir=f"./tokenizer_logs",
        job_name="tokenize_dataset",
        time=args.time,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu_gb=args.mem_per_cpu_gb,
        sbatch_args={"account": "a139", "environment": "datatrove-2508"},
        srun_args={"cpu-bind": "none", "export": "ALL"},
    )

    executor.run()
