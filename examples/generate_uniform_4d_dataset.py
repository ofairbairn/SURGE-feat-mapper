"""Generate a synthetic 2D uniformly-distributed dataset for testing the
Mapper clustering tendency step (Hopkins statistic gate).
One off script.
Saves the flattened (N, 2) dataset as a CSV to runs/generated_datasets/.
"""

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
OUTPUT_PATH = _REPO / "runs" / "generated_datasets" / "uniform_2d.csv"

rng = np.random.default_rng(seed=42)

# Changing this tuple alters the number of data points
#10 * 10 = 100 data points
shape_2d = (10, 10)

# Generate the 2D dataset
data_2d = rng.normal(loc=2.0, scale=1.0, size=shape_2d)

# Verify the size and shape
print(f"Dataset shape: {data_2d.shape}")
print(f"Total number of data points: {data_2d.size}")

# Flatten to a 2D table of points x 2 features for downstream use
points = data_2d.reshape(-1, 2)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
header = "x1,x2"
np.savetxt(OUTPUT_PATH, points, delimiter=",", header=header, comments="")

print(f"Saved {points.shape[0]} points to {OUTPUT_PATH}")
