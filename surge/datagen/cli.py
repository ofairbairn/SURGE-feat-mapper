"""
CLI WRAPPER for converter functions
This module allows users to convert data in their terminal
using the command line interface (CLI) without needing to write Python code.
To convert to CSV/JSON need pandas
To convert to Parquet need pandas and pyarrow
To convert to HDF5 need h5py
"""

#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

from .converters import convert_npy_to_format  # adjust import if function name differs

def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="surge-convert-npy",
        description="Convert .npy features/targets to CSV/Parquet/JSON/HDF5.")
    p.add_argument("features", type=Path, help="Path to features .npy file")
    p.add_argument("--targets", "-t", type=Path, default=None, help="Path to targets .npy file (optional)")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output file path")
    p.add_argument("--format", "-f", choices=["csv","parquet","json","hdf5"], default="csv",
                   help="Output format")
    p.add_argument("--normalize", action="store_true", help="Normalize features to [0,1]")
    p.add_argument("--flatten", action="store_true", help="Flatten image-shaped arrays for tabular formats")
    p.add_argument("--target-name", default="target", help="Column name for targets")
    p.add_argument("--feature-prefix", default=None, help="Prefix for auto-generated feature names (pixel_ or feature_)")
    p.add_argument("--parquet-engine", default="pyarrow", help="Parquet engine (pyarrow or fastparquet)")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if not args.features.exists():
        print(f"ERROR: features file not found: {args.features}", file=sys.stderr)
        return 2
    if args.targets is not None and not args.targets.exists():
        print(f"ERROR: targets file not found: {args.targets}", file=sys.stderr)
        return 2

    # Choose a sensible feature name prefix if not provided
    feature_prefix = args.feature_prefix
    if feature_prefix is None:
        feature_prefix = "pixel" if args.flatten else "feature"

    try:
        out = convert_npy_to_format(
            features_path=str(args.features),
            targets_path=str(args.targets) if args.targets is not None else None,
            output_path=str(args.output),
            format=args.format,
            feature_names=None,
            target_name=args.target_name,
            normalize=args.normalize,
            flatten_images_for_tabular=args.flatten,
            image_shape_meta=True,
            parquet_engine=args.parquet_engine,
            verbose=args.verbose,
        )
        if args.verbose:
            print(f"Saved converted file to: {out}")
        return 0
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        return 1

######### EXAMPLE USAGE OF CLI #########
# cd "C:\Users\Bipo1\Downloads\PPPL Project Materials\SURGE"
# .\.venv\Scripts\Activate.ps1
# python -m surge.datagen.cli \
#   "C:\Users\Bipo1\Downloads\UCI Wine Data\winequality-red.npy" \
#   --output "C:\Users\Bipo1\Downloads\UCI Wine Data\winequality-red.csv" \
#   --format csv --flatten --verbose
########################################

if __name__ == "__main__":
    raise SystemExit(main())