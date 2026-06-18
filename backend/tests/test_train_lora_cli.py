import json
import subprocess
import sys
from pathlib import Path


def test_train_lora_dry_run_writes_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are Cognix."},
                    {"role": "user", "content": "Question: What is semantic search?"},
                    {"role": "assistant", "content": "Semantic search retrieves by meaning."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "adapter"
    result = subprocess.run(
        [
            sys.executable,
            "backend/training/train_lora.py",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--base-model",
            "test-base",
            "--adapter-name",
            "test-adapter",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter_name"] == "test-adapter"
    assert manifest["dataset"]["examples"] == 1
