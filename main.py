import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin
import wandb
import os
import argparse
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.utils import create_dsa_llama_model_from_scratch, create_dsa_llama_model_pretrained, PerformanceTracker, get_model_flops_per_token, load_from_checkpoint
from src.dataset import get_dataloader
from src.losses import ForCausalLMLoss

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
    parser.add_argument("--dataset_offset", type=int, default=0, help="Offset for dataset samples")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="llama-dsa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--load_from_checkpoint", type=str, default=None, help="Path to checkpoint to load model weights from (e.g., warmup checkpoint for sparse training)")

    # Loss config
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--gradient_clipping", type=float, default=float('inf'))

    # Training stage config
    parser.add_argument("--warmup_stage", action="store_true", help="Dense warm-up stage: freeze main model and train only indexer")

    return parser.parse_args()


def train(args):
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=args.gradient_accumulation_steps,
        sync_each_batch=True
    )

    accelerator = Accelerator(
        # mixed_precision="bf16",
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        log_with="wandb",
        # dynamo_backend="inductor",
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
        batch_size=args.batch_size,
        shuffle=True,
        max_samples=args.max_train_samples,
        offset=args.dataset_offset
    )

    accelerator.print(f"✅ Dataset loaded: {len(train_dataloader.dataset)} examples")

    # Create or load model
    if args.load_from_checkpoint:
        accelerator.print(f"📂 Loading model weights from: {args.load_from_checkpoint}")
        
        model = load_from_checkpoint(args.load_from_checkpoint)

        accelerator.print(f"✅ Model loaded from checkpoint")
        if args.warmup_stage:
            accelerator.print("⚠️  Warning: Loading checkpoint but running warmup stage")
        else:
            accelerator.print("🔄 Continuing training in sparse mode")
    else:
        accelerator.print(f"🏗️  Creating new model from: {args.model_name}")
        model = create_dsa_llama_model_pretrained(
            model_path=args.model_name,
            index_top_k=args.index_top_k,
            index_num_heads=args.index_num_heads,
            rope_head_dim=args.rope_head_dim,
            index_head_dim=args.index_head_dim,
        )

    vocab_size = model.config.vocab_size

    if args.warmup_stage:
        accelerator.print("❄️  Freezing main model - training indexer only")
        model.freeze_main_model()

    # Separate optimizers for main model and indexers
    indexer_params = []
    main_model_params = []

    for name, param in model.named_parameters():
        if 'indexer' in name:
            indexer_params.append(param)
        else:
            main_model_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': main_model_params, 'lr': args.learning_rate},
        {'params': indexer_params, 'lr': args.learning_rate}
    ], weight_decay=args.weight_decay)

    # Calculate total training steps for scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps # amount of weight updates the optimizer will do
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # initialize performance tracker
    model_flops_per_token = get_model_flops_per_token(model, args.max_seq_length)
    perf_tracker = PerformanceTracker(warmup_steps=10)

    # Prepare with Accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    ) 

    # Training loop
    global_step = 0
    total_tokens = 0
    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    num_gpus = accelerator.num_processes

    accelerator.print(f"🚀 Starting training for {args.num_epochs} epochs")
    accelerator.print(f"📊 Config: {vars(args)}")
    accelerator.print(f"📊 Model params: {num_params / 1e9:.2f}B | GPUs: {num_gpus}")

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            # NOTE: not usre if you can use multiple backward passes inside accumulate
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False, compute_kl_loss=True, warmup_stage=args.warmup_stage)

                # In sparse training stage, also update main model with CE loss
                if not args.warmup_stage:
                    with accelerator.autocast():
                        ce_loss = ForCausalLMLoss(
                            outputs["logits"],
                            batch["labels"],
                            vocab_size,
                        )
                else:
                    # In warmup stage, compute CE loss for logging only (no backward)
                    with torch.no_grad():
                        ce_loss = ForCausalLMLoss(
                            outputs["logits"],
                            batch["labels"],
                            vocab_size,
                        )

                # NOTE: Doing the sum of the loss should be the same as doing multiple backwards
                with accelerator.autocast():
                    total_kl_loss = sum(outputs["kl_loss"])

                accelerator.backward(ce_loss + total_kl_loss) 
                # NOTE: this doesn't matter as the gradient of sum is always 1 only for myself and both "computational graph paths" (so they don't share any intermediary nodes) are completely separate, so here our gradient shouldn't be ∂ce_loss/∂θ + ∂total_kl_loss/∂θ, but be ∂ce_loss/∂θ and ∂total_kl_loss/∂θ separately


                # Compute gradient norms and clip
                indexer_grad_norm = torch.nn.utils.clip_grad_norm_(
                    indexer_params, max_norm=args.gradient_clipping
                )
                if not args.warmup_stage:
                    main_grad_norm = torch.nn.utils.clip_grad_norm_(
                        main_model_params, max_norm=args.gradient_clipping
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


            # Update performance tracker (Not sure if its correct the way we are measuring)
            batch_tokens = batch["input_ids"].numel()
            perf_metrics = perf_tracker.step(batch_tokens, model_flops_per_token)

            total_tokens += batch_tokens
            global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                log_dict = {
                    "train/ce_loss": ce_loss.detach().item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/total_tokens": total_tokens,
                    "train/indexer_grad_norm": indexer_grad_norm.item(),
                }
                if not args.warmup_stage:
                    log_dict["train/main_grad_norm"] = main_grad_norm.item()

                # Add per-layer KL losses
                for i, layer_kl_loss in enumerate(outputs["kl_loss"]):
                    log_dict[f"train/kl_loss_layer_{i}"] = layer_kl_loss.detach().item()
                log_dict["train/mean_kl_loss"] = sum(
                    kl.detach().item() for kl in outputs["kl_loss"]
                ) / len(outputs["kl_loss"])

                # Add performance metrics if available
                if perf_metrics:
                    log_dict["train/tokens_per_sec"] = perf_metrics.get("tokens_per_second", 0)
                    log_dict["train/tflops_per_gpu"] = perf_metrics.get("tflops_per_device", 0)

                accelerator.log(log_dict, step=global_step)

                perf_str = ""
                if "tflops_per_device" in perf_metrics:
                    perf_str = f" | TFLOPs/GPU: {perf_metrics['tflops_per_device']:.2f}"

                grad_norm_str = f" | Indexer GradNorm: {indexer_grad_norm.item():.2e}"
                if not args.warmup_stage:
                    grad_norm_str += f" | Main GradNorm: {main_grad_norm.item():.2e}"

                accelerator.print(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"CE Loss: {ce_loss.detach().item():.4f} | "
                    f"Mean KL Loss: {sum(kl.detach().item() for kl in outputs['kl_loss']) / len(outputs['kl_loss']):.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}{perf_str}{grad_norm_str}"
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
