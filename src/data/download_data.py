"""Download the raw flotation dataset into data/raw_data/raw.csv."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

DATA_URL = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DATA_URL, help="Source URL of the raw dataset")
    parser.add_argument(
        "--output", default="data/raw_data/raw.csv", help="Destination CSV file"
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(args.url, output)
    print(f"Raw data downloaded to {output}")


if __name__ == "__main__":
    main()
