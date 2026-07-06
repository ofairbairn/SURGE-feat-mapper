"""SURGE-compatible CNN adapter for MNIST-like grayscale image data."""
from __future__ import annotations
#the runner script for this file is located in SURGE/examples/mnist_train_cnn.py
#pedagogical example for Owen
#just run mnist_train_cnn.py to see how this file is used
from pathlib import Path
from typing import Any, Optional

import numpy as np
#imports changed to check for torch availability
from .base import BaseModelAdapter

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

#rewritten CNN
class _MNISTCNN(nn.Module if TORCH_AVAILABLE else object):
    """Small convolutional network for 28x28 grayscale images."""

    def __init__(self, num_classes: int = 10) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MNISTCNNAdapter. Install torch first.")
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) #changed my kernel size from 5 to 3 and padding from 2 to 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) #added second fcn
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1) #not doing the flattening manually now
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# SURGE interacts with this class, it creates my CNN, shapes data, converts labels, trains the model, and handles saving (serialize the model)
class MNISTCNNAdapter(BaseModelAdapter):
    """PyTorch CNN wrapper that trains on flattened or image-shaped MNIST-like data."""

    name = "pytorch.mnist_cnn"
    backend = "pytorch"
    supports_serialization = True
    uses_internal_preprocessing = True
    handles_output_scaling = False

    def __init__(self, **kwargs: Any) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MNISTCNNAdapter. Install torch first.")
        super().__init__(**kwargs)
        self._model = None
        self.training_history: list = []  # Initialize training history tracker

    def _build_model(self, **kwargs: Any) -> Any: # CNN CREATED HERE at _build_model
        num_classes = int(kwargs.get("num_classes", kwargs.get("n_classes", 10)))
        return _MNISTCNN(num_classes=num_classes)

    def _prepare_features(self, X: Any) -> torch.Tensor:  # type: ignore[name-defined] #_prepare_features reshapes raw numpy arrays into tensors
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim == 2 and arr.shape[1] == 28 * 28:
            arr = arr.reshape(-1, 1, 28, 28)
        elif arr.ndim == 3 and arr.shape[1:] == (28, 28):
            arr = arr[:, None, :, :]
        elif arr.ndim == 4 and arr.shape[1:] == (1, 28, 28):
            pass
        else:
            raise ValueError(
                "MNISTCNNAdapter expects flattened 784 features or images shaped (N, 28, 28)"
            )

        arr = arr.astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    def _prepare_targets(self, y: Any) -> torch.Tensor:  # type: ignore[name-defined] #_prepare_targets converts labels into a tensor of int64
        arr = np.asarray(y)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return torch.from_numpy(arr.astype(np.int64))

    def fit( #fit is training loop, it builds the model if it doesn't exist, prepares the data, and trains the model using cross-entropy loss and Adam optimizer
        self,
        X: Any,
        y: Any,
        *,
        X_val: Any = None,
        y_val: Any = None,
        finetune: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self._model is None:
            num_classes = int(kwargs.get("num_classes", kwargs.get("n_classes", 10)))
            self._model = self._build_model(num_classes=num_classes)

        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = y_arr.reshape(-1)
        num_classes = max(int(y_arr.max()) + 1, int(kwargs.get("num_classes", kwargs.get("n_classes", 10))))
        if getattr(self._model, "fc2", None) is None or self._model.fc2.out_features != num_classes:
            self._model = self._build_model(num_classes=num_classes)

        device = kwargs.get("device", self.params.get("device", "cpu"))
        self._model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=float(kwargs.get("learning_rate", self.params.get("learning_rate", 1e-3))))
        batch_size = int(kwargs.get("batch_size", self.params.get("batch_size", 32)))
        epochs = int(kwargs.get("epochs", self.params.get("epochs", 10)))

        X_tensor = self._prepare_features(X).to(device)
        y_tensor = self._prepare_targets(y).to(device)
        train_ds = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        if X_val is not None and y_val is not None:
            val_x = self._prepare_features(X_val).to(device)
            val_y = self._prepare_targets(y_val).to(device)
            val_ds = TensorDataset(val_x, val_y)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        else:
            val_loader = None

        best_state = None
        best_val_loss = float("inf")
        self.training_history = []  # Reset training history for this fit
        self._model.train()
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            train_count = 0
            self._model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                outputs = self._model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() * xb.size(0)
                train_count += xb.size(0)

            avg_train_loss = float(epoch_train_loss / max(1, train_count))

            val_accuracy = None
            val_loss = None
            if val_loader is not None:
                self._model.eval()
                correct = 0
                total = 0
                val_loss_accum = 0.0
                val_count = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        outputs = self._model(xb)
                        val_loss_accum += criterion(outputs, yb).item() * xb.size(0)
                        val_count += xb.size(0)
                        preds = torch.argmax(outputs, dim=1)
                        correct += int((preds == yb).sum().item())
                        total += int(yb.size(0))
                val_loss = float(val_loss_accum / max(1, val_count))
                val_accuracy = float(correct / max(1, total))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                self._model.train()

            # record epoch history for plotting/inspection
            entry = {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_accuracy": val_accuracy, "val_loss": val_loss}
            self.training_history.append(entry)

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        return self

    def predict(self, X: Any) -> Any: #runs inference and returns class labels
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        device = self.params.get("device", "cpu")
        self._model.to(device)
        self._model.eval()
        X_tensor = self._prepare_features(X).to(device)
        with torch.no_grad():
            logits = self._model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds.astype(np.int64)

    def save(self, filepath: str | Path) -> None: #serialize the model
        if self._model is None:
            raise ValueError("Model must be fitted before saving")
        path = Path(filepath)
        payload = {
            "model_state_dict": self._model.state_dict(),
            "num_classes": self._model.fc2.out_features,
            "params": dict(self.params),
        }
        torch.save(payload, path)

    def load(self, filepath: str | Path) -> None: #deserialize the model
        path = Path(filepath)
        payload = torch.load(path, map_location=self.params.get("device", "cpu"))
        num_classes = int(payload.get("num_classes", 10))
        self._model = self._build_model(num_classes=num_classes)
        self._model.load_state_dict(payload["model_state_dict"])
        self._model.to(self.params.get("device", "cpu"))
        self._model.eval()


__all__ = ["MNISTCNNAdapter"]