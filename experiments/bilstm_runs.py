import json
from pathlib import Path

from src.data.dataset import get_loaders
from src.models.rnn import BiLSTM
from src.train import run
from src.utils import set_seed


def main():
    results = {}
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    # run the BiLSTM on output only (RQ1), input only and both (RQ2)
    for mode in ["output", "input", "both"]:
        print(f"\n===== MODE: {mode.upper()} =====")
        train_loader, val_loader, test_loader, vocab, labels = get_loaders(mode)

        # seed before building the model so the starting weights are repeatable
        set_seed(42)
        model = BiLSTM(len(vocab), len(labels))

        metrics = run(
            model, train_loader, val_loader, test_loader, labels,
            epochs=20, lr=0.001, seed=42
        )
        results[mode] = {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        }
        print(results[mode])

        # save after each mode so a crash does not lose finished runs
        with open(output_dir / "bilstm_results.json", "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
