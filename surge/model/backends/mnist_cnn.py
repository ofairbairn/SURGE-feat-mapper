"""MNIST CNN backend for SURGE.

This backend implements the actual PyTorch training loop for the MNIST-like
CNN used by the SURGE examples.  The corresponding adapter lives in
``surge.model.adapters.mnist_cnn`` and handles registry integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None  # type: ignore


class _MNISTCNN(nn.Module if TORCH_AVAILABLE else object):
    """Small convolutional network for 28x28 grayscale images."""

    def __init__(self, num_classes: int = 10) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MNISTCNNModel. Install torch first.")
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MNISTCNNModel:
    """sklearn-compatible CNN backend for MNIST-like classification tasks."""

    def __init__(
        self,
        num_classes: int = 10,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MNISTCNNModel. Install torch first.")
        self.num_classes = int(num_classes)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = int(random_state)
        self.verbose = verbose
        self.log_file = log_file
        self._model: Any = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.training_history: list[dict[str, Any]] = []

    def _build_model(self, num_classes: int) -> _MNISTCNN:
        return _MNISTCNN(num_classes=num_classes)

    @staticmethod
    def _prepare_features(X: Any) -> "torch.Tensor":  # type: ignore[name-defined]
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
                "MNISTCNNModel expects flattened 784 features or images shaped (N, 28, 28)"
            )

        arr = arr.astype(np.float32)
        if float(np.max(arr)) > 2.0:
            arr = arr / 255.0
        return torch.from_numpy(arr)

    def _encode_targets(self, y: Any, *, fit: bool = False) -> np.ndarray:
        y_arr = np.asarray(y).reshape(-1)
        if fit or not hasattr(self.label_encoder, "classes_"):
            return self.label_encoder.fit_transform(y_arr)
        return self.label_encoder.transform(y_arr)

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        y_val: Any = None,
        *,
        finetune: bool = False,
        **kwargs: Any,
    ) -> "MNISTCNNModel":
        del finetune
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        y_encoded = self._encode_targets(y, fit=True)
        num_classes = int(len(self.label_encoder.classes_))
        model = self._model if self._model is not None and self._model.fc2.out_features == num_classes else None
        if model is None:
            model = self._build_model(num_classes=num_classes)
        self._model = model.to(self.device)

        if X_val is not None and y_val is not None:
            y_val_encoded = self._encode_targets(y_val, fit=False)
        else:
            y_val_encoded = None

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self._model.parameters(),
            lr=float(kwargs.get("learning_rate", self.learning_rate)),
        )
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        epochs = int(kwargs.get("epochs", self.epochs))

        X_tensor = self._prepare_features(X).to(self.device)
        y_tensor = torch.from_numpy(np.asarray(y_encoded, dtype=np.int64)).to(self.device)
        train_ds = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        if X_val is not None and y_val_encoded is not None:
            val_x = self._prepare_features(X_val).to(self.device)
            val_y = torch.from_numpy(np.asarray(y_val_encoded, dtype=np.int64)).to(self.device)
            val_ds = TensorDataset(val_x, val_y)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        else:
            val_loader = None

        best_state = None
        best_val_loss = float("inf")
        self.training_history = []
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

            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                }
            )

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> Any:
        if self._model is None or not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        self._model.to(self.device)
        self._model.eval()
        X_tensor = self._prepare_features(X).to(self.device)
        with torch.no_grad():
            logits = self._model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(preds.astype(np.int64))

    def predict_proba(self, X: Any) -> Any:
        if self._model is None or not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        self._model.to(self.device)
        self._model.eval()
        X_tensor = self._prepare_features(X).to(self.device)
        with torch.no_grad():
            logits = self._model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def save(self, filepath: str | Path) -> None:
        if self._model is None:
            raise ValueError("Model must be fitted before saving")
        path = Path(filepath)
        payload = {
            "model_state_dict": self._model.state_dict(),
            "num_classes": self._model.fc2.out_features,
            "params": {
                "num_classes": self.num_classes,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "device": str(self.device),
                "random_state": self.random_state,
            },
            "label_classes": self.label_encoder.classes_.tolist(),
            "training_history": list(self.training_history),
        }
        torch.save(payload, path)

    def load(self, filepath: str | Path) -> None:
        path = Path(filepath)
        payload = torch.load(path, map_location=self.device)
        num_classes = int(payload.get("num_classes", payload.get("params", {}).get("num_classes", 10)))
        self._model = self._build_model(num_classes=num_classes)
        self._model.load_state_dict(payload["model_state_dict"])
        self._model.to(self.device)
        self._model.eval()
        self.is_fitted = True
        self.training_history = list(payload.get("training_history", []))
        classes = payload.get("label_classes")
        if classes is None:
            self.label_encoder.fit(np.arange(num_classes))
        else:
            self.label_encoder.classes_ = np.asarray(classes)


__all__ = ["MNISTCNNModel"]