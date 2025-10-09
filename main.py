# train.py
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from model import CustomLLM, CustomLLMConfig
import wandb
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
assert os.getenv("HF_TOKEN")


from transformers import LlamaConfig, LlamaForCausalLM

def create_llama_model_from_scratch():
    # Use official LLaMA 3.2 1B config (architecture only)
    config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")

    # Optional: inspect it
    print("Model config:", config)

    # Build model from config (NO WEIGHTS LOADED)
    model = LlamaForCausalLM(config)
    
    # Optional: Freeze embeddings for scratch training (optional)
    # model.get_input_embeddings().weight.requires_grad = False

    return model


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
def train():
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4,
        log_with="wandb"
    )
    accelerator.init_trackers("custom_llm_training")

    # ✅ Use correct tokenizer with 128k vocab
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token

    # ✅ Match model config to tokenizer
    config = CustomLLMConfig(
        vocab_size=len(tokenizer),  # Should be 128256 for Llama-3
        hidden_size=2048,
        num_hidden_layers=16,       # Adjust to match JSON
        num_attention_heads=32,
        intermediate_size=8192
    )

    # Dataset and DataLoader
    train_dataset = TextDataset("data.txt", tokenizer, max_length=512)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # Model from scratch
    # model = CustomLLM(config)
    model = create_llama_model_from_scratch()

    # Optimizer + LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # Prepare with Accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # Validate data tokens
    validate_token_ids(train_dataloader, model)

    model.train()
    for epoch in range(3):
        for step, batch in enumerate(train_dataloader):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss, loss_components = custom_loss(outputs["logits"], batch["labels"], alpha=0.05)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.log({
                    "loss": loss.item(),
                    "ce_loss": loss_components["ce_loss"],
                    "entropy": loss_components["entropy"],
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch
                })
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
