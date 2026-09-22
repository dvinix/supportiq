"""Production SFT QLoRA training script for SupportIQ.

Supports both dry-run estimation (CPU) and full GPU execution (Kaggle, Colab, SageMaker).
"""

import argparse
import sys
from pathlib import Path

from supportiq.core.logger import get_logger
from supportiq.training.config import TrainingConfig, load_training_config

logger = get_logger(__name__)


def dry_run_summary(config: TrainingConfig, data_dir: Path | str = "data/processed") -> None:
    """Print architectural and budget dry-run summary without requiring GPU or PyTorch."""
    train_file = Path(data_dir) / "train.jsonl"

    num_train = 0
    if train_file.exists():
        with open(train_file, encoding="utf-8") as f:
            num_train = sum(1 for _ in f)

    batch_size = config.training.per_device_train_batch_size
    grad_accum = config.training.gradient_accumulation_steps
    effective_batch = batch_size * grad_accum
    epochs = config.training.num_train_epochs
    total_steps = (num_train // effective_batch) * epochs if num_train > 0 else 0

    print("=" * 60)
    print("           SUPPORTIQ QLORA TRAINING DRY RUN")
    print("=" * 60)
    print(f"Base Model:             {config.model.name}")
    print(f"Dev Alternative:        {config.model.dev_model_name}")
    print(f"Precision:              4-bit NF4 ({config.quantization.bnb_4bit_compute_dtype})")
    print(f"LoRA Rank (r):          {config.lora.r}")
    print(f"LoRA Alpha:             {config.lora.lora_alpha}")
    print(f"LoRA Dropout:           {config.lora.lora_dropout}")
    print(f"Target Modules:         {', '.join(config.lora.target_modules[:4])} ...")
    print(f"Max Sequence Length:    {config.training.max_seq_length} tokens")
    print("-" * 60)
    print(f"Training Rows:          {num_train:,} examples")
    print(f"Per-Device Batch Size:  {batch_size}")
    print(f"Gradient Accumulation:  {grad_accum}")
    print(f"Effective Batch Size:   {effective_batch}")
    print(f"Epochs:                 {epochs}")
    print(f"Estimated Total Steps:  ~{total_steps:,} steps")
    print("Estimated VRAM (0.6B):  ~2.5 GB (Fits easily on free T4 16GB)")
    print("Estimated VRAM (4B):    ~8.5 GB (Fits easily on free T4 16GB)")
    print("Estimated Kaggle Cost:  $0.00 (Free GPU)")
    print("=" * 60)


def run_training_gpu(
    config: TrainingConfig,
    train_path: str,
    val_path: str,
    output_dir: str,
    resume_from_checkpoint: str | None = None,
) -> None:
    """Execute real GPU training with TRL SFTTrainer and PEFT."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as e:
        logger.error("Missing GPU deep learning dependencies: %s", e)
        print("Error: PyTorch and Hugging Face GPU packages are required for real training.")
        print("Please run on a GPU environment (Kaggle, Colab, or AWS SageMaker).")
        sys.exit(1)

    logger.info("Initializing GPU training for model: %s", config.model.name)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 1. 4-bit Quantization Config (BitsAndBytes)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # 2. Load Base Model in 4-bit
    logger.info("Loading base model: %s in 4-bit NF4...", config.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Prepare Model for LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
        target_modules=config.lora.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Load SFT Conversational Dataset
    logger.info("Loading SFT JSONL datasets...")
    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        evaluation_strategy=config.training.evaluation_strategy,
        eval_steps=config.training.eval_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        optim="paged_adamw_8bit",
        seed=config.training.seed,
        report_to="none",
    )

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        dataset_text_field="messages",
        max_seq_length=config.training.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Starting SFT training loop...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 7. Save Final Adapter
    final_adapter_path = Path(output_dir) / "final_adapter"
    logger.info("Saving trained adapter to %s...", final_adapter_path)
    model.save_pretrained(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))
    print(f"\nSUCCESS: LoRA adapter saved to {final_adapter_path}")


def main() -> None:
    """CLI entry point for training launcher."""
    parser = argparse.ArgumentParser(description="SupportIQ QLoRA SFT Fine-Tuning Engine")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and step estimates without running")
    parser.add_argument("--model-name", type=str, default=None, help="Override base model name")
    parser.add_argument("--train-file", type=str, default="data/processed/train.jsonl", help="Path to train JSONL")
    parser.add_argument("--val-file", type=str, default="data/processed/val.jsonl", help="Path to validation JSONL")
    parser.add_argument("--output-dir", type=str, default="artifacts/qlora_checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    config = load_training_config(args.config)
    if args.model_name:
        config.model.name = args.model_name

    if args.dry_run:
        dry_run_summary(config)
        return

    run_training_gpu(
        config=config,
        train_path=args.train_file,
        val_path=args.val_file,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
