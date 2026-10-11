
import json
from pathlib import Path

from src.data.dataset import get_loaders
from src.models.cnn import TextCNN
from src.train import run
from src.utils import set_seed


def main():
    results = {}
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare input-only and combined input-output text
    for mode in ["input", "both"]:
        print(f"\n===== MODE: {mode.upper()} =====")

        train_loader, val_loader, test_loader, vocab, labels = get_loaders(
            mode=mode,
            max_len=256,
            batch_size=64
        )

        results[mode] = {}

        # Train and evaluate CNN
        print(f"\nTraining CNN with {mode} text...")

        set_seed(42)

        model = TextCNN(
            vocab_size=len(vocab),
            num_classes=len(labels)
        )

        metrics = run(
            model,
            train_loader,
            val_loader,
            test_loader,
            labels,
            epochs=10,
            lr=0.001,
            seed=42
        )

        results[mode]["CNN"] = {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"])
        }

        print(f"CNN accuracy: {metrics['accuracy']:.4f}")
        print(f"CNN macro F1: {metrics['macro_f1']:.4f}")

        # Save results after each completed run
        with open(output_dir / "rq2_results.json", "w") as f:
            json.dump(results, f, indent=4)

    print("\n===== FINAL RQ2 RESULTS =====")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
