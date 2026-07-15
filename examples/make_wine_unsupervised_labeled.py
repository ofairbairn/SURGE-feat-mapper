"""Build wine unsupervised datasets with optional label preservation.

Usage:
    python examples/make_wine_unsupervised_labeled.py \
        --red "C:/path/winequality-red-comma.csv" \
        --white "C:/path/winequality-white-comma.csv"

Outputs next to each source CSV:
- *-unsupervised.csv           (drops label column)
- *-unsupervised-labeled.csv   (keeps label column for viz color_by=label)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _write_variants(source_path: Path, *, label_col: str) -> None:
    df = pd.read_csv(source_path)
    stem = source_path.with_suffix("")

    unlabeled_path = Path(str(stem) + "-unsupervised.csv")
    labeled_path = Path(str(stem) + "-unsupervised-labeled.csv")

    if label_col in df.columns:
        df.drop(columns=[label_col]).to_csv(unlabeled_path, index=False)
    else:
        df.to_csv(unlabeled_path, index=False)

    df.to_csv(labeled_path, index=False)

    print(f"wrote {unlabeled_path}")
    print(f"wrote {labeled_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create unsupervised wine CSV variants with and without labels."
    )
    parser.add_argument("--red", type=Path, required=True, help="Path to red wine CSV")
    parser.add_argument("--white", type=Path, required=True, help="Path to white wine CSV")
    parser.add_argument(
        "--label-col",
        default="quality",
        help="Column name to preserve in *-unsupervised-labeled.csv",
    )
    args = parser.parse_args()

    for src in (args.red, args.white):
        src = src.resolve()
        if not src.exists():
            raise FileNotFoundError(f"CSV not found: {src}")
        _write_variants(src, label_col=args.label_col)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
