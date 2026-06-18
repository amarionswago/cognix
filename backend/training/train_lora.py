"""LoRA supervised fine-tuning entrypoint for Cognix.

Dry-run mode is dependency-light and validates the dataset/manifest. Actual
training requires the optional `backend[training]` dependency group and suitable
hardware.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.finetuning import build_training_manifest, register_planned_lora_artifact, write_training_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or validate a Cognix LoRA adapter.")
    parser.add_argument("--dataset", required=True, type=Path, help="Chat-style SFT JSONL dataset.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Adapter output directory.")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter-name", default="cognix-lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate data and write manifest without training.")
    args = parser.parse_args()

    manifest = build_training_manifest(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        base_model=args.base_model,
        adapter_name=args.adapter_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    manifest_path = write_training_manifest(manifest, args.output_dir)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Examples: {manifest['dataset']['examples']}")

    if args.dry_run:
        return 0

    register_planned_lora_artifact(manifest)
    run_lora_training(args.dataset, args.output_dir, args.base_model, args.epochs, args.learning_rate, args.batch_size)
    return 0


def run_lora_training(
    dataset_path: Path,
    output_dir: Path,
    base_model: str,
    epochs: float,
    learning_rate: float,
    batch_size: int,
) -> None:
    """Run an actual LoRA SFT job when optional training dependencies exist."""
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: .venv/bin/python -m pip install -e 'backend[training]'"
        ) from exc

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
