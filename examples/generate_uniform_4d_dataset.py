"""Generate a synthetic 4D uniformly-distributed dataset for testing the
Mapper clustering tendency step (Hopkins statistic gate).

Saves the flattened (N, 4) dataset as a CSV to runs/generated_datasets/.
"""

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
OUTPUT_PATH = _REPO / "runs" / "generated_datasets" / "uniform_4d.csv"

rng = np.random.default_rng(seed=42)

# Changing this tuple alters the number of data points
# 5 * 5 * 5 * 8 = 1,000 total elements
shape_4d = (5, 5, 5, 8)

# Generate the 4D dataset
data_4d = rng.uniform(low=0.0, high=1.0, size=shape_4d)

# Verify the size and shape
print(f"Dataset shape: {data_4d.shape}")
print(f"Total number of data points: {data_4d.size}")

# Flatten to a 2D table of points x 4 features for downstream use
points = data_4d.reshape(-1, 4)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
header = "x1,x2,x3,x4"
np.savetxt(OUTPUT_PATH, points, delimiter=",", header=header, comments="")

print(f"Saved {points.shape[0]} points to {OUTPUT_PATH}")
