"""Count the target classes in train.csv and save a table and chart."""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_splits" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "distribution_results" / "class_balance"


def main():
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Count examples of each target class.
    class_counts = df["label"].value_counts().sort_index()
    class_table = class_counts.to_frame("count")
    class_table["percent"] = class_counts / len(df) * 100
    class_table.to_csv(OUTPUT_DIR / "class_balance.csv", encoding="utf-8-sig")
    print("\nClass balance:")
    print(class_table.round(2).to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts.plot.bar(ax=ax, color="#367a9c", rot=0)
    ax.set(xlabel="Sport", ylabel="Number of training responses",
           title="Training class balance")
    ax.bar_label(ax.containers[0])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_balance.png", dpi=160)
    plt.close(fig)

    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
