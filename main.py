import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import argparse
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.dsa_llama_model import DSALlamaForCausalLM, DSALlamaConfig
from src.utils import create_dsa_llama_model_from_scratch, create_dsa_llama_model_pretrained

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
    parser.add_argument("--data_file", type=str, default="data.txt")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")

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


# ===== CUSTOM LOSS FUNC =====
def custom_loss(logits, labels, alpha=0.05):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    ce_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )

    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean()

    return ce_loss + alpha * entropy, {"ce_loss": ce_loss.item(), "entropy": entropy.item()}

# ===== DATASET SETUP =====
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, "r") as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.tokenizer(
            self.lines[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = text["input_ids"][0]
        labels = input_ids.clone()
        labels[text["attention_mask"][0] == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": text["attention_mask"][0],
            "labels": labels
        }

# ===== VALIDATION =====
def validate_token_ids(dataloader, model):
    vocab_size = model.config.vocab_size
    print(f"🔍 Validating tokens against vocab_size = {vocab_size}")
    for step, batch in enumerate(dataloader):
        if batch["input_ids"].max() >= vocab_size:
            print("🚨 Bad token ID found!")
            print(batch["input_ids"].max())
            raise RuntimeError("Token ID > vocab_size")
        if step > 5:  # Just check first few batches
            break
    print("✅ Token validation passed")

# ===== TRAIN LOOP =====
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset and DataLoader
    train_dataset = TextDataset(args.data_file, tokenizer, max_length=args.max_seq_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

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
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Prepare with Accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # Validate data tokens
    validate_token_ids(train_dataloader, model)

    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    global_step = 0
    model.train()

    accelerator.print(f"🚀 Starting training for {args.num_epochs} epochs")
    accelerator.print(f"📊 Config: {vars(args)}")

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss, loss_components = custom_loss(
                    outputs["logits"],
                    batch["labels"],
                    alpha=args.loss_alpha
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
                    "train/ce_loss": loss_components["ce_loss"],
                    "train/entropy": loss_components["entropy"],
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
