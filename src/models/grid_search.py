"""Find the best Gradient Boosting hyperparameters with a grid search."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed_data")
    parser.add_argument("--output", default="models/best_params.pkl")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", default="r2")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    X_train = pd.read_csv(input_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(input_dir / "y_train.csv").squeeze("columns")

    search = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid=PARAM_GRID,
        cv=args.cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
    )
    search.fit(X_train, y_train)

    result = {
        "model": "GradientBoostingRegressor",
        "params": search.best_params_,
        "cv_score": float(search.best_score_),
        "scoring": args.scoring,
        "cv_folds": args.cv,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, output)

    print(f"Best params: {result['params']}")
    print(f"Best cross-validated {args.scoring}: {result['cv_score']:.4f}")


if __name__ == "__main__":
    main()
