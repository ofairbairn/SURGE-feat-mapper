# MNIST Data Conversion and SURGE Workflow Guide

## Quick Start

### Step 1: Convert Your MNIST .npy Files

The converter is already built into SURGE. Run this command:

```bash
python examples/mnist_data_converter_example.py
```

This will automatically:
- Read your `.npy` files from `C:\Users\Bipo1\Downloads\MNIST Data\`
- Convert to both CSV and Parquet formats
- Save to `data/datasets/mnist_converted/`

**Output:**
- `mnist_train.csv` (1.7 MB) and `mnist_train.parquet` (0.9 MB)
- `mnist_test.csv` (0.2 MB) and `mnist_test.parquet` (0.5 MB)

### Step 2: Run SURGE with the Converted Data

```bash
surge run examples/configs/mnist_classification.yaml
```

This will:
- Load training data from the Parquet file
- Auto-detect that the last column (`digit`) is the target
- Train Random Forest and CNN models
- Save results to `runs/mnist_classification_v1/`

### Step 3: View Results

Results are saved in `runs/mnist_classification_v1/`:

```
runs/mnist_classification_v1/
├── metrics.json                           # Train/val/test accuracy per model
├── workflow_summary.json                  # Full run summary
├── spec.yaml                              # Your config (reproducible)
├── models/
│   ├── Random Forest.joblib               # Trained model
│   └── CNN (MNIST).joblib                 # Trained model
├── predictions/
│   ├── Random Forest_train.parquet        # Training predictions
│   ├── Random Forest_val.parquet
│   ├── Random Forest_test.parquet
│   ├── CNN (MNIST)_train.parquet
│   ├── CNN (MNIST)_val.parquet
│   └── CNN (MNIST)_test.parquet
├── scalers/
│   ├── inputs.joblib
│   └── outputs.joblib
└── training_history_CNN_MNIST_.json       # Training loss/accuracy curves
```

## Advanced Usage

### Use the Converter Programmatically

```python
from surge.datagen import convert_mnist_npy_to_csv, convert_mnist_npy_to_parquet
from pathlib import Path

# Convert to CSV
csv_path = convert_mnist_npy_to_csv(
    features_path="C:/Users/Bipo1/Downloads/MNIST Data/mnist_train_features.npy",
    targets_path="C:/Users/Bipo1/Downloads/MNIST Data/mnist_train_targets.npy",
    output_path="my_data/mnist_train.csv",
    target_name="digit",
)

# Or convert to Parquet (faster + smaller)
parquet_path = convert_mnist_npy_to_parquet(
    features_path="C:/Users/Bipo1/Downloads/MNIST Data/mnist_train_features.npy",
    targets_path="C:/Users/Bipo1/Downloads/MNIST Data/mnist_train_targets.npy",
    output_path="my_data/mnist_train.parquet",
    target_name="digit",
)
```

### Enable Hyperparameter Optimization

Edit `examples/configs/mnist_classification.yaml` and uncomment the HPO section:

```yaml
models:
  - key: pytorch.mnist_cnn
    name: "CNN (MNIST)"
    params:
      epochs: 10
      batch_size: 32
      learning_rate: 1e-3
      device: "cpu"
    hpo:
      enabled: true
      n_trials: 20
      metric: "val_accuracy"
      direction: "maximize"
      search_space:
        batch_size: [8, 16, 32, 64]
        learning_rate: [1e-4, 1e-3, 1e-2]
        epochs: [5, 10, 15]
```

Then run:
```bash
surge run examples/configs/mnist_classification.yaml
```

### Visualize Results

After running, generate plots:

```bash
surge viz runs/mnist_classification_v1/
```

This generates:
- `inference_comparison_output_*.png` — Predicted vs Ground Truth scatter plots
- `training_plot_CNN_MNIST__loss_accuracy.png` — Training curves (loss + accuracy)

## Data Format Reference

### Input Format (Your .npy Files)

- `mnist_train_features.npy`: Shape (1000, 28, 28) — pixel values 0–255
- `mnist_train_targets.npy`: Shape (1000,) — digit labels 0–9
- `mnist_test_features.npy`: Shape (100, 28, 28)
- `mnist_test_targets.npy`: Shape (100,)

### Output Format (Converted CSV)

| pixel_0 | pixel_1 | pixel_2 | ... | pixel_783 | digit |
|---------|---------|---------|-----|-----------|-------|
| 0       | 10      | 20      | ... | 5         | 5     |
| 150     | 200     | 210     | ... | 50        | 3     |

- 784 pixel columns (28×28)
- 1 target column (`digit`)
- ~1000 rows (training) + ~100 rows (test)

## File Locations

| File | Path |
|------|------|
| Converter script | `examples/mnist_data_converter_example.py` |
| Converter API | `surge/datagen/converters.py` |
| Config template | `examples/configs/mnist_classification.yaml` |
| Converted data | `data/datasets/mnist_converted/` |
| Run outputs | `runs/mnist_classification_v1/` |

## Troubleshooting

### "No such file or directory" when running converter

**Solution:** Update the path in `examples/mnist_data_converter_example.py`:

```python
MNIST_DATA_DIR = Path("C:\\Users\\Bipo1\\Downloads\\MNIST Data")
```

### SURGE says "No input or output columns detected"

**Solution:** The converter automatically names columns as `pixel_0` ... `pixel_783` and `digit` (target).
SURGE should auto-detect these. If not, check that the CSV/Parquet loads correctly:

```python
import pandas as pd
df = pd.read_csv("data/datasets/mnist_converted/mnist_train.csv")
print(df.columns)  # Should show pixel_0, ..., pixel_783, digit
print(df.shape)    # Should be (1000, 785)
```

### CNN accuracy is low

**Solution:** 
- Increase `epochs` in the config (try 20 instead of 10)
- Increase `batch_size` (try 64 instead of 32)
- Reduce `learning_rate` (try 5e-4 instead of 1e-3)
- Use more training data (1000 samples may be limited)

## Summary

You now have:

1. ✅ **Converter utility** (`surge/datagen/converters.py`) — Converts .npy to CSV/Parquet
2. ✅ **Example script** (`examples/mnist_data_converter_example.py`) — One-command conversion
3. ✅ **SURGE config** (`examples/configs/mnist_classification.yaml`) — Ready to run
4. ✅ **Converted data** (`data/datasets/mnist_converted/`) — Ready for training

Run this to see everything in action:

```bash
# Convert data
python examples/mnist_data_converter_example.py

# Train models
surge run examples/configs/mnist_classification.yaml

# View plots
surge viz runs/mnist_classification_v1/
```
