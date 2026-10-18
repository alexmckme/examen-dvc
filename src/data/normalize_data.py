"""Standardize the training and test features.

The scaler is fitted on the training set only to avoid data leakage and is
saved so that the exact same transformation can be reused at inference time.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed_data")
    parser.add_argument("--output-dir", default="data/processed_data")
    parser.add_argument(
        "--scaler-output", default="models/scaler.pkl", help="Path of the fitted scaler"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    X_train = pd.read_csv(input_dir / "X_train.csv")
    X_test = pd.read_csv(input_dir / "X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train_scaled.to_csv(output_dir / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(output_dir / "X_test_scaled.csv", index=False)

    scaler_output = Path(args.scaler_output)
    scaler_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_output)

    print(f"Normalization done: {X_train_scaled.shape[1]} features scaled")


if __name__ == "__main__":
    main()
