"""
Evaluate base Llama vs DSA Llama using lm-evaluation-harness

IMPORTANT: This script must be run from the Sparse-Attention-Zoo directory
so that the DSA model code (src/dsa_llama_model.py, src/indexer.py, etc.)
is available for loading checkpoints.
"""
import argparse
import numpy as np
import os
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import json
# Ensure we can import the DSA model code
from src.dsa_llama_model import DSALlamaForCausalLM, DSALlamaConfig

# Register DSA model classes with transformers so AutoModel can load them
AutoConfig.register("dsa_llama", DSALlamaConfig)
AutoModelForCausalLM.register(DSALlamaConfig, DSALlamaForCausalLM)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Llama models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (HF name or checkpoint)")
    parser.add_argument("--model_type", type=str, choices=["base", "dsa"], default="base")
    parser.add_argument("--tasks", type=str, default="hellaswag,arc_easy,winogrande", help="Comma-separated tasks")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str, default="eval_results.json")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading {'base' if args.model_type == 'base' else 'DSA'} model: {args.model_path}")

    # Pass model path as string to HFLM for proper accelerate support
    # lm-eval will use AutoModelForCausalLM.from_pretrained() which will
    # find our registered DSA model class via the config's model_type
    lm_obj = HFLM(
        pretrained=args.model_path,
        dtype="bfloat16",
        batch_size=args.batch_size,
        # parallelize=True # This is for dividing the model across gpus naively
    )

    # Run tasks
    tasks = args.tasks.split(",")

    # For RULER (it seems it is not possible to generate examples of 2048, there are no possible examples that can be generated and it gets stuck in a loop of num_docs > increment when both are equal in qa_utils.py)
    metadata = {"pretrained": args.model_path, "max_seq_lengths": [4096]}

    results = evaluator.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        batch_size=args.batch_size,
        metadata=metadata
    )

    if lm_obj.accelerator.is_main_process:
        if results is None:
            print("Warning: No results returned from evaluation")
            return

        # Save and print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for task in tasks:
            if task in results["results"]:
                print(f"\n{task}:")
                for metric, value in results["results"][task].items():
                    if not metric.startswith("alias"):
                        print(f"  {metric}: {value}")

        def handle_non_serializable(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            elif isinstance(o, set):
                return list(o)
            else:
                return str(o)

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        with open(args.output_path, "w") as f:
            json.dump(
                results, f, indent=2, default=handle_non_serializable, ensure_ascii=False
            )
        print(f"\nResults saved to: {args.output_path}")

if __name__ == "__main__":
    main()
