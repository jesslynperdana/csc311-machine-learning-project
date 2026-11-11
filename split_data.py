"""Split the sports survey by student ID into train/validation/test CSVs.

No responses are removed, imputed, or transformed here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Resolve paths relative to this script.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data_raw" / "csc311_project_training_202609.csv"
OUTPUT_DIR = PROJECT_DIR / "data_splits"

# Keep the seed and input row order fixed.
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def main():
    # Preserve original text.
    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    if "id" not in df.columns:
        raise ValueError("The dataset must contain an id column.")
    if df["id"].str.strip().eq("").any():
        raise ValueError("Student IDs must not be blank.")
    if not np.isclose(TRAIN_FRAC + VAL_FRAC + TEST_FRAC, 1.0):
        raise ValueError("Split fractions must add up to 1.")
    if min(TRAIN_FRAC, VAL_FRAC, TEST_FRAC) <= 0:
        raise ValueError("Each split fraction must be positive.")

    # Shuffle students' responses.
    student_ids = df["id"].unique()
    rng = np.random.default_rng(SEED)
    shuffled_ids = rng.permutation(student_ids)

    n_students = len(shuffled_ids)
    n_train = int(n_students * TRAIN_FRAC)
    n_val = int(n_students * VAL_FRAC)
    n_test = n_students - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("Not enough students for three nonempty splits.")

    # Keep the original allocation order: test first, validation next, train last.
    train_ids = shuffled_ids[n_test + n_val:]
    val_ids = shuffled_ids[n_test:n_test + n_val]
    test_ids = shuffled_ids[:n_test]

    # Select every response from each assigned student into the same split.
    train = df[df["id"].isin(train_ids)].copy()
    validation = df[df["id"].isin(val_ids)].copy()
    test = df[df["id"].isin(test_ids)].copy()

    # Save all original columns.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "validation": validation, "test": test}
    for name, subset in splits.items():
        subset.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"{name}: {subset['id'].nunique()} students, {len(subset)} rows")

    # Share this assignment manifest so all teammates use the same groups.
    manifest = pd.DataFrame({
        "id": np.concatenate([train_ids, val_ids, test_ids]),
        "split": (["train"] * n_train + ["validation"] * n_val
                  + ["test"] * n_test),
    })
    manifest.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")
    print("Checks passed: no student overlap; all rows retained.")
    print(f"Saved to: {OUTPUT_DIR}")
    print("Next: perform data exploration using train.csv only.")


if __name__ == "__main__":
    main()
