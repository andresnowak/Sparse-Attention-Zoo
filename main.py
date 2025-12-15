import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin, DistributedType
import wandb
import os
import argparse
import time
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.utils import (
    create_dsa_llama_model_from_scratch,
    create_dsa_llama_model_pretrained,
    PerformanceTracker,
    get_model_flops_per_token,
    load_from_checkpoint,
    get_model_size_breakdown,
    TrainingMetrics,
)
from src.dataset import get_dataloader
from src.losses import ForCausalLMLoss
from src.token_selection_tracker import TokenSelectionTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train Llama with Dynamic Sparse Attention")
    
    # Model config
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--index_top_k", type=int, default=2048)
    parser.add_argument("--index_num_heads", type=int, default=16)
    parser.add_argument("--rope_head_dim", type=int, default=32)
    parser.add_argument("--index_head_dim", type=int, default=64)
    
    # Training config
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--global_batch_size", type=int, required=True, help="Global batch size across all GPUs (must be divisible by micro_batch_size * num_gpus as we are assuming DDP and Zero)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total steps used for linear warmup")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    
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

    # Baseline
    parser.add_argument("--baseline_experiment", action="store_true", default=False, help="We do an experiment with the model without the indexer so as to have a baseline of the loss")

    # Token selection tracking
    parser.add_argument("--track_token_selection", action="store_true", default=False, help="Track which tokens are selected and log heatmaps to wandb")
    parser.add_argument("--track_log_every", type=int, default=100, help="Log token selection heatmaps every N steps")
    parser.add_argument("--track_save_every", type=int, default=10_000, help="Save token selection tracker every N steps")

    return parser.parse_args()


def train(args):
    # Get number of processes from environment variables set by accelerate launcher
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate gradient accumulation steps from global batch size
    # NOTE: here we are assuming DDP (and Zero if its used)
    global_batch_size = args.global_batch_size
    if global_batch_size % (args.micro_batch_size * num_processes) != 0:
        raise ValueError(f"Global batch size ({global_batch_size}) must be divisible by (micro_batch_size * num_processes) = ({args.micro_batch_size} * {num_processes})")

    gradient_accumulation_steps = global_batch_size // (args.micro_batch_size * num_processes)

    # Update accelerator with gradient accumulation
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=gradient_accumulation_steps,
        sync_each_batch=True # use less memory
    )

    # NOTE: Split batches is false by default in accelerator, so batch size of dataloader is considered as an MBS (so each rank in DP will use this MBS and )
    accelerator = Accelerator(
        # mixed_precision="bf16",
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        log_with="wandb",
        dynamo_backend="inductor",
    )
    accelerator.print(f"Global batch size: {global_batch_size}")
    accelerator.print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # Initialize wandb
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.wandb_run_name}}
    )
    wandb_tracker = accelerator.get_tracker("wandb")

    with accelerator.main_process_first():
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
        batch_size=args.micro_batch_size,
        shuffle=True,
        max_samples=args.max_train_samples,
        offset=args.dataset_offset
    )

    accelerator.print(f"✅ Dataset loaded: {len(train_dataloader.dataset)} examples")

    # Create or load model
    if args.load_from_checkpoint and not args.baseline_experiment:
        accelerator.print(f"📂 Loading model weights from: {args.load_from_checkpoint}")
        
        model = load_from_checkpoint(args.load_from_checkpoint)

        accelerator.print(f"✅ Model loaded from checkpoint")
        if args.warmup_stage:
            accelerator.print("⚠️  Warning: Loading checkpoint but running warmup stage")
        else:
            accelerator.print("🔄 Continuing training in sparse mode")
    elif args.baseline_experiment:
        accelerator.print(f"Running baseline for: {args.model_name}")
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16
        )
    else:
        accelerator.print(f"🏗️  Creating new model from pretrained: {args.model_name}")
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

    # Calculate total training steps for scheduler (total steps is per rank)
    total_steps = len(train_dataloader) * args.num_epochs // gradient_accumulation_steps # amount of weight updates the optimizer will do
    warmup_steps = int(args.warmup_ratio * total_steps)

    accelerator.print(f"Total training steps: {total_steps} and total warmup steps: {warmup_steps}")
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1, # Start percentage of base LR
                end_factor=1.0, # Finish at base LR
                total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps, # we remove the warmup steps
                eta_min=args.min_lr
            )
        ],
        milestones=[warmup_steps] # when to switch from one scheduler to the next one
    )

    # Initialize token selection tracker if requested
    token_tracker = None
    if args.track_token_selection and not args.baseline_experiment:
        accelerator.print("📊 Initializing token selection tracker")
        if hasattr(model.config, 'index_top_k'):
            token_tracker = TokenSelectionTracker(
                top_k=model.config.index_top_k,
                layers=[0, 1, 2, 8, 9, 13, 14, 15],
                save_dir=os.path.join(args.save_dir, "token_tracker"),
                accelerator=accelerator
            )

    # Prepare with Accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # initialize performance tracker and metrics aggregator (after prepare to get correct step count)
    # Use unwrapped model to access config
    unwrapped_model = accelerator.unwrap_model(model)
    model_flops_per_token = get_model_flops_per_token(unwrapped_model, args.max_seq_length)
    perf_tracker = PerformanceTracker(warmup_steps=10)

    # Calculate actual number of steps this process will see
    actual_steps_per_process = len(train_dataloader) * args.num_epochs
    metrics_tracker = TrainingMetrics(accelerator, perf_tracker, model_flops_per_token, total_steps=actual_steps_per_process)

    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    num_gpus = accelerator.num_processes

    accelerator.print(f"🚀 Starting training for {args.num_epochs} epochs")
    accelerator.print(f"📊 Config: {vars(args)}")
    accelerator.print(f"📊 Model params: {num_params / 1e9:.2f}B | GPUs: {num_gpus}")

    # Print detailed model size breakdown (use unwrapped model)
    accelerator.print(get_model_size_breakdown(unwrapped_model))

    indexer_grad_norm = None
    main_grad_norm = None

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            iter_start_time = time.perf_counter()

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch, use_cache=False, compute_kl_loss=not args.baseline_experiment, warmup_stage=args.warmup_stage) # The baseline model doesn't have this options

                # In sparse training stage, also update main model with CE loss
                if not args.warmup_stage or args.baseline_experiment:
                    with accelerator.autocast():
                        ce_loss = ForCausalLMLoss(
                            outputs["logits"],
                            batch["labels"],
                            vocab_size,
                        )
                else:
                    # In warmup stage, compute CE loss for logging only (no backward)
                    with torch.no_grad():
                        with accelerator.autocast():
                            ce_loss = ForCausalLMLoss(
                                outputs["logits"],
                                batch["labels"],
                                vocab_size,
                            )

                # NOTE: Remember backward in accelerate does a mean by default of the reduce of the gradients over all ranks (mean of ranks)
                if not args.baseline_experiment:
                    # NOTE: Doing the sum of the loss should be the same as doing multiple backwards
                    total_kl_loss = outputs["total_kl_loss"]

                    with accelerator.autocast():
                        total_loss = ce_loss + total_kl_loss
                    accelerator.backward(total_loss)

                    # NOTE: this doesn't matter as the gradient of sum is always 1 only for myself and both "computational graph paths" (so they don't share any intermediary nodes) are completely separate, so here our gradient shouldn't be ∂ce_loss/∂θ + ∂total_kl_loss/∂θ, but be ∂ce_loss/∂θ and ∂total_kl_loss/∂θ separately
                else:
                    accelerator.backward(ce_loss)


                # Compute gradient norms and clip
                if accelerator.sync_gradients:
                    if not args.baseline_experiment:
                        indexer_grad_norm = accelerator.clip_grad_norm_(
                            indexer_params, max_norm=args.gradient_clipping
                        )
                    if not args.warmup_stage or args.baseline_experiment:
                        main_grad_norm = accelerator.clip_grad_norm_(
                            main_model_params, max_norm=args.gradient_clipping
                        )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # We are outside accumulate (so we are being called even if a training step is not being done on the accumulate)
            # Compute iteration time
            iter_time = time.perf_counter() - iter_start_time

            # Aggregate metrics and log (all ranks must call this - contains collective operations)
            metrics = metrics_tracker.step(
                batch=batch,
                ce_loss=ce_loss,
                outputs=outputs,
                epoch=epoch,
                scheduler=scheduler,
                iter_time=iter_time,
                indexer_grad_norm=indexer_grad_norm,
                main_grad_norm=main_grad_norm,
                baseline_experiment=args.baseline_experiment,
                warmup_stage=args.warmup_stage,
                log_every=args.log_every,
            )

            # Track token selections if enabled
            if token_tracker is not None and not args.baseline_experiment and metrics["global_step"] % args.track_log_every == 0:
                indexer_scores = outputs.get("indexer_scores")
                if indexer_scores is not None and len(indexer_scores) > 0:
                    token_tracker.record_selections(metrics["global_step"], indexer_scores, wandb_tracker)

            # Save token tracker periodically
            if token_tracker is not None and metrics["global_step"] > 0 and metrics["global_step"] % args.track_save_every == 0:
                token_tracker.save()


            # Checkpointing
            if metrics["global_step"] % args.save_every == 0:
                checkpoint_path = os.path.join(args.save_dir, f"checkpoint-{metrics['global_step']}")
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
