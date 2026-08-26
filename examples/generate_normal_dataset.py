"""Generate a synthetic 1D normally-distributed dataset for testing the
Mapper clustering tendency step (Hopkins statistic gate).
One off script.
Saves the flattened (N, 1) dataset as a CSV to runs/generated_datasets/.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Setup script output paths
_REPO = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
OUTPUT_PATH = _REPO / "runs" / "generated_datasets" / "all_scaler_comparison.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(seed=42)
n_samples = 1000

# 1. Generate two features with drastically different scales and shapes
# x1: Normal distribution
x1 = rng.normal(loc=-6.0, scale=1.0, size=n_samples)

# x2: Exponential distribution offset at +50 (skewed with a large scale and outliers)
x2 = rng.exponential(scale=15.0, size=n_samples) + 50
#x2 = rng.normal(loc=6.0, scale = 1.0, size=n_samples)
#x2 = rng.uniform(low=30.0, high=40.0, size=n_samples)

# here I meanually add extreme outliers in both x1
#x2[100:116]= 10.0
# Combine into shape (n_samples, 2)
X = np.column_stack([x1, x2])

# 2. Define scalers
scalers = {
    "Unscaled (Original) Exp+outliers": None,
    "StandardScaler Exp+outliers": StandardScaler(),
    "RobustScaler Exp+outliers": RobustScaler(),
    "MinMaxScaler Exp+outliers": MinMaxScaler()
}

# 3. Plot the KDE distributions for x1 and x2
fig, axes = plt.subplots(4, 1, figsize=(4, 8))
axes = axes.flatten()

for ax, (title, scaler) in zip(axes, scalers.items()):
    # Scale data if scaler is provided
    X_scaled = X if scaler is None else scaler.fit_transform(X)

    # Plot KDE density curves for both features
    sns.kdeplot(X_scaled[:, 0], ax=ax, label="x1 (Gaussian)", fill=True, alpha=0.3, color="tab:blue")
    sns.kdeplot(X_scaled[:, 1], ax=ax, label="x2 (Exponential)", fill=True, alpha=0.3, color="tab:orange")
    
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Value Range")
    ax.set_ylabel("Density")
    # ax.axvline(x=np.percentile(x1, 25), color="green", linestyle="--", alpha=0.6)
    # ax.axvline(x=np.percentile(x1, 75), color="green", linestyle="--", alpha=0.6)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()

print(f"Saved comparison plot to: {OUTPUT_PATH}")