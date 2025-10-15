import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb
import os
import argparse
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.utils import create_dsa_llama_model_from_scratch, create_dsa_llama_model_pretrained, PerformanceTracker, get_model_flops_per_token
from src.dataset import get_dataloader

# Load environment variables from .env
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
assert os.getenv("HF_TOKEN")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Llama with Dynamic Sparse Attention")
    
    # Model config
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--index_top_k", type=int, default=2048)
    parser.add_argument("--index_num_heads", type=int, default=16)
    parser.add_argument("--rope_head_dim", type=int, default=32)
    parser.add_argument("--index_head_dim", type=int, default=64)
    
    # Training config
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # Data config
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="HuggingFace dataset config")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--text_column", type=str, default="text", help="Column name containing text data")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples to use")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="llama-dsa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=1000)

    # Loss config
    parser.add_argument("--weight_decay", type=float, default=0.1)

    return parser.parse_args()


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


def train(args):
    # Initialize accelerator with args

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # NOTE: Fix because indexer doesn't get gradients, because we don't have KL divergence loss. I don't know if this is also necessary to activate Distributed Data parallelism

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
        kwargs_handlers=[kwargs]
    )

    # Initialize wandb
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.wandb_run_name}}
    )

    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare dataset
    accelerator.print(f"📚 Loading dataset: {args.dataset_name}")

    train_dataloader = get_dataloader(
        accelerator=accelerator,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        batch_size=16,
        shuffle=True,
        max_samples=args.max_train_samples
    )

    accelerator.print(f"✅ Dataset loaded: {len(train_dataloader)} examples")

    # Create model with args
    model = create_dsa_llama_model_from_scratch(
        model_path=args.model_name,
        index_top_k=args.index_top_k,
        index_num_heads=args.index_num_heads,
        rope_head_dim=args.rope_head_dim,
        index_head_dim=args.index_head_dim,
    )

    # Optimizer + LR scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Calculate total training steps for scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps # amount of weight updates the optimizer will do
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # initialize performance tracker
    model_flops_per_token = get_model_flops_per_token(model, args.max_seq_length)
    perf_tracker = PerformanceTracker(warmup_steps=10)

    # Prepare with Accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    ) # here is where the sharding happens

    # Training loop
    global_step = 0
    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    num_gpus = accelerator.num_processes

    accelerator.print(f"🚀 Starting training for {args.num_epochs} epochs")
    accelerator.print(f"📊 Config: {vars(args)}")
    accelerator.print(f"📊 Model params: {num_params / 1e9:.2f}B | GPUs: {num_gpus}")

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = cross_entropy_loss(
                    outputs["logits"],
                    batch["labels"],
                )

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Update performance tracker (Not sure if its correct the way we are measuring)
            batch_tokens = batch["input_ids"].numel()
            perf_metrics = perf_tracker.step(batch_tokens, model_flops_per_token)

            global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                log_dict = {
                    "train/loss": loss.detach().item(),
                    "train/ce_loss": loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }

                # Add performance metrics if available
                if perf_metrics:
                    log_dict["train/tokens_per_sec"] = perf_metrics.get("tokens_per_second", 0)
                    log_dict["train/tflops_per_gpu"] = perf_metrics.get("tflops_per_device", 0)

                accelerator.log(log_dict, step=global_step)

                perf_str = ""
                if "tflops_per_device" in perf_metrics:
                    perf_str = f" | TFLOPs/GPU: {perf_metrics['tflops_per_device']:.2f}"

                accelerator.print(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}{perf_str}"
                )

            # Checkpointing
            if global_step % args.save_every == 0:
                checkpoint_path = os.path.join(args.save_dir, f"checkpoint-{global_step}")
                accelerator.print(f"💾 Saving checkpoint to {checkpoint_path}")
                accelerator.wait_for_everyone()

                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    checkpoint_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save
                )

                if accelerator.is_main_process:
                    tokenizer.save_pretrained(checkpoint_path)

    # Final save
    final_path = os.path.join(args.save_dir, "final_model")
    accelerator.print(f"💾 Saving final model to {final_path}")
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        final_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save
    )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_path)

    accelerator.end_training()
    accelerator.print("✅ Training complete!")

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
