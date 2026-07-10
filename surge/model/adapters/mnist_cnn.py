"""Adapter registering ``pytorch.mnist_cnn`` in the SURGE model registry."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ...hpc import ResourceProfile

_MNIST_PROFILE = ResourceProfile(
    name="pytorch.mnist_cnn",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="MNIST-style CNN for variable-size image classification (e.g., 28x28, 64x64).",
)


class MNISTCNNAdapter(BaseModelAdapter):
    """PyTorch CNN wrapper for MNIST-like classification tasks."""

    name = "pytorch.mnist_cnn"
    backend = "pytorch"
    uses_internal_preprocessing = True
    resource_profile = _MNIST_PROFILE
    task_type = "classification"

    default_params: dict[str, Any] = {
        "num_classes": 10,
        "input_size": (28, 28),
        "input_channels": 1,
        "hidden_channels": (16, 32),
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "random_state": 42,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        from surge.model.backends.mnist_cnn import MNISTCNNModel

        params = dict(self.default_params)
        params.update(kwargs)
        return MNISTCNNModel(**params)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        X_val: Any = None,
        y_val: Any = None,
        finetune: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self._model is None and not finetune:
            self._model = self._build_model(**self.params)
        return self._model.fit(X, y, X_val=X_val, y_val=y_val, finetune=finetune, **kwargs)

    def predict(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict_proba(X)

    def save(self, filepath: Any) -> None:
        if self._model is None:
            raise ValueError("Model must be fitted before saving")
        self._model.save(filepath)

    def load(self, filepath: Any) -> None:
        from surge.model.backends.mnist_cnn import MNISTCNNModel

        self._model = MNISTCNNModel()
        self._model.load(filepath)


__all__ = ["MNISTCNNAdapter"]