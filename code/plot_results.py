"""
plot_results.py
Read the training log CSV produced by train_0323_v2.py and output:
    1. training_curves.png  – loss and accuracy curves (2x2 grid)
    2. lr_curve.png         – learning rate schedule across epochs

Usage:
    python plot_results.py --log model/se_resnet50_v2_log.csv
    python plot_results.py --log model/se_resnet50_v2_log.csv --out_dir figures/
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_log(log_path: str) -> pd.DataFrame:
    df = pd.read_csv(log_path)
    # normalise column names (strip whitespace just in case)
    df.columns = df.columns.str.strip()
    return df


def save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)


# ── Plot 1: training curves ────────────────────────────────────────────────────

def plot_training_curves(df: pd.DataFrame, out_path: str) -> None:
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # ── Loss ──
    ax = axes[0]
    ax.plot(epochs, df["train_loss"], label="Train Loss", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, df["val_loss"],   label="Val Loss",   color="#F44336", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Accuracy ──
    ax = axes[1]
    # convert to percentage if values are in [0, 1]
    train_acc = df["train_acc"] * 100 if df["train_acc"].max() <= 1.0 else df["train_acc"]
    val_acc   = df["val_acc"]   * 100 if df["val_acc"].max()   <= 1.0 else df["val_acc"]
    ax.plot(epochs, train_acc, label="Train Acc", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_acc,   label="Val Acc",   color="#F44336", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # # annotate best val acc
    # best_epoch = df.loc[df["val_acc"].idxmax(), "epoch"]
    # best_acc   = val_acc.max()
    # ax.annotate(
    #     f"Best: {best_acc:.2f}% @ ep{int(best_epoch)}",
    #     xy=(best_epoch, best_acc),
    #     xytext=(best_epoch + max(len(df) * 0.05, 1), best_acc - 3),
    #     arrowprops=dict(arrowstyle="->", color="gray"),
    #     fontsize=9, color="gray",
    # )

    fig.tight_layout()
    save(fig, out_path)


# ── Plot 2: learning rate curve ───────────────────────────────────────────────

def plot_lr_curve(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["epoch"], df["lr"], color="#4CAF50", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    save(fig, out_path)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    best_row = df.loc[df["val_acc"].idxmax()]
    val_acc_pct = best_row["val_acc"] * 100 if best_row["val_acc"] <= 1.0 else best_row["val_acc"]
    print("\n── Training Summary ─────────────────────────────")
    print(f"  Total epochs       : {int(df['epoch'].max())}")
    print(f"  Best val accuracy  : {val_acc_pct:.2f}%  (epoch {int(best_row['epoch'])})")
    print(f"  Best val loss      : {best_row['val_loss']:.4f}")
    print(f"  Final train acc    : {df['train_acc'].iloc[-1]*100:.2f}%")
    print(f"  Final val acc      : {df['val_acc'].iloc[-1]*100:.2f}%")
    print("─────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from log CSV")
    parser.add_argument("--log",     type=str, required=True,
                        help="Path to the log CSV (e.g. model/se_resnet50_v2_log.csv)")
    parser.add_argument("--out_dir", type=str, default="figures",
                        help="Directory to save output figures (default: figures/)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.log):
        raise FileNotFoundError(f"Log file not found: {args.log}")

    df = load_log(args.log)
    print(f"Loaded {len(df)} epochs from {args.log}")
    print_summary(df)

    plot_training_curves(df, out_path=os.path.join(args.out_dir, "training_curves.png"))
    plot_lr_curve(df,        out_path=os.path.join(args.out_dir, "lr_curve.png"))


if __name__ == "__main__":
    main()