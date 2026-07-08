import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training history
history_file = Path("runs/runs/mnist_classification_v2/training_history_CNN_MNIST_.json")
with open(history_file) as f:
    history = json.load(f)

# Extract metrics
epochs = list(range(1, len(history) + 1))
train_loss = [h.get("train_loss") for h in history]
val_accuracy = [h.get("val_accuracy") for h in history]
val_loss = [h.get("val_loss") for h in history]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Loss
ax1.plot(epochs, train_loss, marker='o', linewidth=2, markersize=6, color='#1f77b4')
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Training Loss", fontsize=12)
ax1.set_title("CNN MNIST Training Loss Over 10 Epochs", fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs)

# Plot 2: Validation Accuracy
ax2.plot(epochs, val_accuracy, marker='s', color='#2ca02c', linewidth=2, markersize=6)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Validation Accuracy", fontsize=12)
ax2.set_title("CNN MNIST Validation Accuracy Over 10 Epochs", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)
ax2.set_ylim([0.8, 1.0])

plt.tight_layout()
plt.savefig("mnist_training_summary.png", dpi=150, bbox_inches='tight')
print("✅ Saved: mnist_training_summary.png")

# Print summary
print(f"\n📊 Training Summary:")
print(f"  Initial Loss: {train_loss[0]:.4f}")
print(f"  Final Loss:   {train_loss[-1]:.4f}")
print(f"  Loss Reduction: {(train_loss[0] - train_loss[-1]) / train_loss[0] * 100:.1f}%")
print(f"  Peak Val Accuracy: {max(val_accuracy):.2%} (Epoch {val_accuracy.index(max(val_accuracy)) + 1})")
print(f"  Final Val Accuracy: {val_accuracy[-1]:.2%}")
