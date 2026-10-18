"""Evaluate the trained model and save the metrics and predictions."""

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed_data")
    parser.add_argument("--model", default="models/trained_model.pkl")
    parser.add_argument("--predictions-output", default="data/prediction.csv")
    parser.add_argument("--metrics-output", default="metrics/scores.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    X_test = pd.read_csv(input_dir / "X_test_scaled.csv")
    y_test = pd.read_csv(input_dir / "y_test.csv").squeeze("columns")

    model = joblib.load(args.model)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    scores = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "n_test_samples": int(len(y_test)),
    }

    predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]

    predictions_output = Path(args.predictions_output)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_output, index=False)

    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print(f"Evaluation: {scores}")


if __name__ == "__main__":
    main()
