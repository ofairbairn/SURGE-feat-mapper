"""MNIST CNN backend for SURGE.

This backend implements the actual PyTorch training loop for the MNIST-like
CNN used by the SURGE examples.  The corresponding adapter lives in
``surge.model.adapters.mnist_cnn`` and handles registry integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    from tqdm.auto import trange

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    trange = None  # type: ignore[assignment]
    TQDM_AVAILABLE = False

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
    """Small convolutional network that supports variable input spatial sizes."""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        hidden_channels: tuple[int, ...] = (16, 32),
        dropout2d_prob: float = 0.2,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MNISTCNNModel. Install torch first.")
        if len(hidden_channels) < 2:
            raise ValueError("hidden_channels must have at least two entries")
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2d = nn.Dropout2d(p=float(dropout2d_prob))

        # Adaptive pooling makes the model input-size agnostic.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        fc_in = hidden_channels[-1]
        self.fc1 = nn.Linear(fc_in, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MNISTCNNModel:
    """sklearn-compatible CNN backend for MNIST-like classification tasks."""

    def __init__(
        self,
        num_classes: int = 10,
        input_size: tuple[int, int] | int = (28, 28),
        input_channels: int = 1,
        hidden_channels: tuple[int, ...] = (16, 32),
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        dropout2d: float = 0.2,
        use_lr_scheduler: bool = True,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 2,
        lr_scheduler_min_lr: float = 1e-6,
        early_stopping: bool = True,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 1e-4,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MNISTCNNModel. Install torch first.")
        self.num_classes = int(num_classes)
        if isinstance(input_size, int):
            self.input_size = (int(input_size), int(input_size))
        else:
            self.input_size = (int(input_size[0]), int(input_size[1]))
        self.input_channels = int(input_channels)
        self.hidden_channels = tuple(int(h) for h in hidden_channels)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout2d = float(dropout2d)
        self.use_lr_scheduler = bool(use_lr_scheduler)
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.lr_scheduler_min_lr = float(lr_scheduler_min_lr)
        self.early_stopping = bool(early_stopping)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = int(random_state)
        self.verbose = verbose
        self.log_file = log_file
        self._model: Any = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.training_history: list[dict[str, Any]] = []

    def _build_model(self, num_classes: int) -> _MNISTCNN:
        return _MNISTCNN(
            num_classes=num_classes,
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            dropout2d_prob=self.dropout2d,
        )

    def _prepare_features(self, X: Any) -> "torch.Tensor":  # type: ignore[name-defined]
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        h, w = self.input_size
        expected_flat = self.input_channels * h * w

        if arr.ndim == 2 and arr.shape[1] == expected_flat:
            arr = arr.reshape(-1, self.input_channels, h, w)
        elif arr.ndim == 3 and self.input_channels == 1 and arr.shape[1:] == (h, w):
            arr = arr[:, None, :, :]
        elif arr.ndim == 4 and arr.shape[1:] == (self.input_channels, h, w):
            pass
        elif arr.ndim == 4 and arr.shape[-1] == self.input_channels and arr.shape[1:3] == (h, w):
            # Channel-last input (N, H, W, C) -> channel-first (N, C, H, W).
            arr = np.transpose(arr, (0, 3, 1, 2))
        else:
            raise ValueError(
                f"MNISTCNNModel expects flattened {expected_flat} features, "
                f"(N, H, W) with input_channels=1, or images shaped "
                f"(N, C, H, W)/(N, H, W, C) matching input_size={self.input_size} "
                f"and input_channels={self.input_channels}"
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
        learning_rate = float(kwargs.get("learning_rate", self.learning_rate))
        weight_decay = float(kwargs.get("weight_decay", self.weight_decay))
        optimizer = optim.AdamW( #switched optimizer to adamw for weight decay of 0.01
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        epochs = int(kwargs.get("epochs", self.epochs))
        show_progress = bool(kwargs.get("show_progress", self.verbose))
        use_lr_scheduler = bool(kwargs.get("use_lr_scheduler", self.use_lr_scheduler))
        lr_scheduler_factor = float(kwargs.get("lr_scheduler_factor", self.lr_scheduler_factor))
        lr_scheduler_patience = max(1, int(kwargs.get("lr_scheduler_patience", self.lr_scheduler_patience)))
        lr_scheduler_min_lr = float(kwargs.get("lr_scheduler_min_lr", self.lr_scheduler_min_lr))
        use_early_stopping = bool(kwargs.get("early_stopping", self.early_stopping))
        early_stopping_patience = max(1, int(kwargs.get("early_stopping_patience", self.early_stopping_patience)))
        early_stopping_min_delta = float(kwargs.get("early_stopping_min_delta", self.early_stopping_min_delta))

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

        scheduler = None
        if val_loader is not None and use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                min_lr=lr_scheduler_min_lr,
            )

        best_state = None
        best_val_loss = float("inf")
        no_improve_epochs = 0
        self.training_history = []
        self._model.train()
        epoch_iterator = (
            trange(epochs, desc="MNISTCNN Training", unit="epoch", disable=not show_progress)
            if show_progress and TQDM_AVAILABLE
            else range(epochs)
        )
        for epoch in epoch_iterator:
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
                if val_loss < (best_val_loss - early_stopping_min_delta):
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                self._model.train()

            if scheduler is not None and val_loss is not None:
                scheduler.step(val_loss)

            current_lr = float(optimizer.param_groups[0]["lr"])

            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                }
            )

            if show_progress and TQDM_AVAILABLE and hasattr(epoch_iterator, "set_postfix"):
                progress_metrics = {"train_loss": f"{avg_train_loss:.4f}"}
                if val_loss is not None:
                    progress_metrics["val_loss"] = f"{val_loss:.4f}"
                if val_accuracy is not None:
                    progress_metrics["val_acc"] = f"{val_accuracy * 100:.2f}%"
                progress_metrics["lr"] = f"{current_lr:.2e}"
                epoch_iterator.set_postfix(progress_metrics)

            if (
                val_loader is not None
                and use_early_stopping
                and no_improve_epochs >= early_stopping_patience
            ):
                break

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

    def analyze_classification(
        self,
        X: Any,
        y: Any,
        *,
        class_names: Optional[Sequence[str]] = None,
        top_k: int = 3,
        print_report: bool = True,
    ) -> dict[str, Any]:
        """Summarize class accuracy, common confusions, and mistaken samples."""
        y_true = np.asarray(y).reshape(-1)
        y_pred = np.asarray(self.predict(X)).reshape(-1)
        y_prob = np.asarray(self.predict_proba(X))
        labels = np.asarray(self.label_encoder.classes_)
        self.label_encoder.transform(y_true)
        return self.analyze_predictions(
            y_true,
            y_pred,
            y_prob=y_prob,
            labels=labels,
            class_names=class_names,
            top_k=top_k,
            print_report=print_report,
        )

    @staticmethod
    def analyze_predictions(
        y_true: Any,
        y_pred: Any,
        *,
        y_prob: Any = None,
        labels: Optional[Sequence[Any]] = None,
        class_names: Optional[Sequence[str]] = None,
        top_k: int = 3,
        print_report: bool = True,
    ) -> dict[str, Any]:
        """Summarize labels and predictions, including mistaken sample indices."""
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must contain the same number of samples")

        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        labels = np.asarray(labels)
        if y_prob is not None:
            y_prob = np.asarray(y_prob, dtype=float)
            expected_shape = (len(y_true), len(labels))
            if y_prob.shape != expected_shape:
                raise ValueError(
                    f"y_prob must have shape {expected_shape}; received {y_prob.shape}"
                )
        if class_names is not None and len(class_names) != len(labels):
            raise ValueError(
                f"class_names must contain {len(labels)} names, one for each class label"
            )
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        names = [str(label) for label in labels] if class_names is None else list(class_names)
        per_class = []
        for class_index, (label, name) in enumerate(zip(labels, names)):
            class_indices = np.where(y_true == label)[0]
            correct = int(np.sum(y_pred[class_indices] == label))
            total = int(len(class_indices))
            accuracy = 100.0 * correct / total if total else None
            class_ece = None
            if y_prob is not None:
                from ...metrics import expected_calibration_error

                binary_targets = (y_true == label).astype(int)
                class_ece = expected_calibration_error(
                    binary_targets,
                    y_prob[:, class_index],
                )
            per_class.append(
                {
                    "label": label.item() if isinstance(label, np.generic) else label,
                    "name": name,
                    "correct": correct,
                    "total": total,
                    "accuracy": accuracy,
                    "ece": class_ece,
                }
            )

        observed_classes = [result for result in per_class if result["total"] > 0]
        hardest_class = min(observed_classes, key=lambda result: result["accuracy"], default=None)

        confusions = []
        for true_label, true_name in zip(labels, names):
            for predicted_label, predicted_name in zip(labels, names):
                if predicted_label == true_label:
                    continue
                sample_indices = np.where(
                    (y_true == true_label) & (y_pred == predicted_label)
                )[0].tolist()
                if sample_indices:
                    confusions.append(
                        {
                            "true_label": (
                                true_label.item()
                                if isinstance(true_label, np.generic)
                                else true_label
                            ),
                            "true_name": true_name,
                            "predicted_label": (
                                predicted_label.item()
                                if isinstance(predicted_label, np.generic)
                                else predicted_label
                            ),
                            "predicted_name": predicted_name,
                            "count": len(sample_indices),
                            "sample_indices": sample_indices,
                        }
                    )
        confusions.sort(key=lambda result: (-result["count"], result["true_name"], result["predicted_name"]))
        top_confusions = confusions[:top_k]
        misclassified_indices = np.where(y_true != y_pred)[0].tolist()

        analysis = {
            "per_class": per_class,
            "hardest_class": hardest_class,
            "top_confusions": top_confusions,
            "misclassified_indices": misclassified_indices,
        }
        if print_report:
            for result in per_class:
                accuracy_text = "N/A" if result["accuracy"] is None else f'{result["accuracy"]:.2f} %'
                print(f'Accuracy of {result["name"]}: {accuracy_text}')
                if result["ece"] is not None:
                    print(
                        f'ECE of {result["name"]} (one-vs-rest): '
                        f'{result["ece"]:.4f} ({result["ece"] * 100:.2f} %)'
                    )
            if hardest_class is not None:
                print(
                    f'Hardest class: {hardest_class["name"]} '
                    f'({hardest_class["accuracy"]:.2f} %)'
                )
            for confusion in top_confusions:
                print(
                    f'Most confused: {confusion["true_name"]} -> '
                    f'{confusion["predicted_name"]}: {confusion["count"]} samples '
                    f'(indices: {confusion["sample_indices"]})'
                )
        return analysis

    def save(self, filepath: str | Path) -> None:
        if self._model is None:
            raise ValueError("Model must be fitted before saving")
        path = Path(filepath)
        payload = {
            "model_state_dict": self._model.state_dict(),
            "num_classes": self._model.fc2.out_features,
            "params": {
                "num_classes": self.num_classes,
                "input_size": list(self.input_size),
                "input_channels": self.input_channels,
                "hidden_channels": list(self.hidden_channels),
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "dropout2d": self.dropout2d,
                "use_lr_scheduler": self.use_lr_scheduler,
                "lr_scheduler_factor": self.lr_scheduler_factor,
                "lr_scheduler_patience": self.lr_scheduler_patience,
                "lr_scheduler_min_lr": self.lr_scheduler_min_lr,
                "early_stopping": self.early_stopping,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
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
        params = payload.get("params", {})
        input_size = params.get("input_size")
        if input_size is not None:
            self.input_size = (int(input_size[0]), int(input_size[1]))
        self.input_channels = int(params.get("input_channels", self.input_channels))
        hidden_channels = params.get("hidden_channels")
        if hidden_channels is not None:
            self.hidden_channels = tuple(int(h) for h in hidden_channels)
        self.weight_decay = float(params.get("weight_decay", self.weight_decay))
        self.dropout2d = float(params.get("dropout2d", self.dropout2d))
        self.use_lr_scheduler = bool(params.get("use_lr_scheduler", self.use_lr_scheduler))
        self.lr_scheduler_factor = float(params.get("lr_scheduler_factor", self.lr_scheduler_factor))
        self.lr_scheduler_patience = int(params.get("lr_scheduler_patience", self.lr_scheduler_patience))
        self.lr_scheduler_min_lr = float(params.get("lr_scheduler_min_lr", self.lr_scheduler_min_lr))
        self.early_stopping = bool(params.get("early_stopping", self.early_stopping))
        self.early_stopping_patience = int(params.get("early_stopping_patience", self.early_stopping_patience))
        self.early_stopping_min_delta = float(params.get("early_stopping_min_delta", self.early_stopping_min_delta))

        num_classes = int(payload.get("num_classes", params.get("num_classes", 10)))
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