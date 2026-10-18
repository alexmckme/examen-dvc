"""Split the raw dataset into training and test sets."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "silica_concentrate"
NON_FEATURE_COLUMNS = ["date"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/raw_data/raw.csv", help="Raw dataset CSV file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_data",
        help="Directory where the split datasets are written",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(
        f"Split done: {len(X_train)} train rows, {len(X_test)} test rows, "
        f"{X_train.shape[1]} features"
    )


if __name__ == "__main__":
    main()
