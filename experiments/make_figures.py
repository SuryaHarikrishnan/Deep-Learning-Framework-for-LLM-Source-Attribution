import json

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

T = "results/tables/"
F = "results/figures/"
COLORS = {"TgF-IDF + LR": "#7F7F7F", "CNN": "#2F5496", "BiLSTM": "#C55A11"}
CHANCE = 100 / 6  # six balanced classes


def load_accuracies():
    # collect test accuracy (in %) for every model and input mode from the saved result files
    base = pd.read_csv(T + "baseline_tfidf.csv").set_index("mode")["accuracy"] * 100
    cnn_out = json.load(open(T + "cnn_results.json"))["test_accuracy"] * 100
    cnn_rq2 = json.load(open(T + "rq2_results.json"))
    bilstm = json.load(open(T + "bilstm_results.json"))
    return {
        "TF-IDF + LR": {m: base[m] for m in ["input", "output", "both"]},
        "CNN": {"input": cnn_rq2["input"]["CNN"]["accuracy"] * 100, "output": cnn_out,
                "both": cnn_rq2["both"]["CNN"]["accuracy"] * 100},
        "BiLSTM": {m: bilstm[m]["accuracy"] * 100 for m in ["input", "output", "both"]},
    }


def label_bars(ax, bars, fmt="{:.1f}"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=9)


def fig_rq1(acc):
    # RQ1: output-only accuracy per model
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(acc)
    bars = ax.bar(names, [acc[n]["output"] for n in names], color=[COLORS[n] for n in names])
    ax.axhline(CHANCE, color="red", linestyle="--", linewidth=1, label="Chance (16.7%)")
    label_bars(ax, bars)
    ax.set_ylim(0, 100); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("RQ1: Identifying the LLM from its response")
    ax.legend(); fig.tight_layout(); fig.savefig(F + "rq1_output_only.png", dpi=150); plt.close(fig)


def fig_rq2(acc):
    # RQ2: input only vs output only vs both, grouped by model
    fig, ax = plt.subplots(figsize=(8, 4.2))
    modes = [("input", "Input only"), ("output", "Output only"), ("both", "Input + Output")]
    w = 0.26
    for i, name in enumerate(acc):
        xs = [j + (i - 1) * w for j in range(3)]
        bars = ax.bar(xs, [acc[name][m] for m, _ in modes], w, label=name, color=COLORS[name])
        label_bars(ax, bars)
    ax.axhline(CHANCE, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in modes])
    ax.set_ylim(0, 100); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("RQ2: Does the prompt help? (red line = chance)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88)); fig.tight_layout()
    fig.savefig(F + "rq2_input_vs_output.png", dpi=150); plt.close(fig)


def fig_rq4_ablation():
    # RQ4: how accurate is each signal on its own, or after removing a signal
    df = pd.read_csv(T + "rq4_ablation.csv").sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(df["setting"], df["accuracy"] * 100, color="#2F5496")
    ax.axvline(CHANCE, color="red", linestyle="--", linewidth=1, label="Chance (16.7%)")
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}",
                va="center", fontsize=9)
    ax.set_xlim(0, 100); ax.set_xlabel("Test accuracy (%)")
    ax.set_title("RQ4: Which signals identify the LLM?")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(F + "rq4_ablation.png", dpi=150); plt.close(fig)


def fig_rq4_style():
    # RQ4: average style features per LLM
    df = pd.read_csv(T + "rq4_style_features.csv", index_col=0)
    cols = [("chars", "Avg. characters"), ("punctuation", "Avg. punctuation marks"),
            ("vocab_variety", "Vocabulary variety")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (c, title) in zip(axes, cols):
        ax.bar(df.index, df[c], color="#2F5496")
        ax.set_title(title, fontsize=11); ax.tick_params(axis="x", rotation=45)
    fig.suptitle("RQ4: Style features of each LLM's responses (training set)")
    fig.tight_layout(); fig.savefig(F + "rq4_style_features.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    acc = load_accuracies()
    fig_rq1(acc); fig_rq2(acc); fig_rq4_ablation(); fig_rq4_style()
    print("saved figures to", F)
