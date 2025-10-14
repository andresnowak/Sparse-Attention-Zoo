import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import argparse
from dotenv import load_dotenv
from transformers import AutoTokenizer
from datasets import load_dataset

from src.dsa_llama_model import DSALlamaForCausalLM, DSALlamaConfig
from src.utils import create_dsa_llama_model_from_scratch, create_dsa_llama_model_pretrained
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
    parser.add_argument("--num_index_heads", type=int, default=16)
    parser.add_argument("--rope_head_dim", type=int, default=32)
    parser.add_argument("--index_hidden_size", type=int, default=1024)
    
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
    parser.add_argument("--loss_alpha", type=float, default=0.05, help="Entropy regularization weight")
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
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
    )

    # Initialize wandb with all config
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
        num_index_heads=args.num_index_heads,
        rope_head_dim=args.rope_head_dim,
        index_hidden_size=args.index_hidden_size
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

    # Prepare with Accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    ) # here is where the sharding happens

    # Training loop
    global_step = 0
    model.train()

    accelerator.print(f"🚀 Starting training for {args.num_epochs} epochs")
    accelerator.print(f"📊 Config: {vars(args)}")

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

            global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/ce_loss": loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": global_step
                }, step=global_step)
                accelerator.print(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
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
