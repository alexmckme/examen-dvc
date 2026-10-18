"""Train the final model using the best hyperparameters found by GridSearch."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed_data")
    parser.add_argument("--best-params", default="models/best_params.pkl")
    parser.add_argument("--output", default="models/trained_model.pkl")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    X_train = pd.read_csv(input_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(input_dir / "y_train.csv").squeeze("columns")

    search_result = joblib.load(args.best_params)
    model = GradientBoostingRegressor(
        random_state=42, **search_result["params"]
    )
    model.fit(X_train, y_train)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)

    print(f"Model trained with {model.n_estimators} estimators and saved to {output}")


if __name__ == "__main__":
    main()
