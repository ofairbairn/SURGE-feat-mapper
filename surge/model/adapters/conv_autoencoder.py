"""Adapter registering ``pytorch.conv_autoencoder``."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from ...hpc import ResourceProfile
from ..base import BaseModelAdapter, ModelInfo


class ConvAutoencoderAdapter(BaseModelAdapter):
    """SURGE adapter for convolutional image reconstruction and encoding."""

    name = "pytorch.conv_autoencoder"
    backend = "pytorch"
    task_type = "unsupervised"
    uses_internal_preprocessing = False
    handles_output_scaling = False
    resource_profile = ResourceProfile(
        name=name,
        supports_cpu=True,
        supports_gpu=True,
        worker_semantics="dataloader_workers",
        notes="num_workers maps to torch DataLoader workers.",
    )
    _INFO = ModelInfo(
        architecture=(
            "Conv2d encoder and ConvTranspose2d decoder joined by a dense "
            "vector bottleneck."
        ),
        use_cases=[
            "Image reconstruction",
            "Spatial representation learning",
            "Image anomaly detection",
        ],
        not_for=["Non-grid tabular data without a meaningful image reshape"],
        strengths=["Preserves spatial structure", "Accepts flattened or NCHW image batches"],
        weaknesses=["Input shape and convolution geometry must be specified in advance"],
        references=["Hinton & Salakhutdinov (2006); convolutional autoencoder extension."],
    )
    default_params: Dict[str, Any] = {
        "latent_dim": 8,
        "channels": (32, 64, 128),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "learning_rate": 1e-3,
        "n_epochs": 150,
        "batch_size": 128,
        "random_state": 42,
        "verbose": False,
    }

    def __init__(
        self,
        *,
        input_shape: Tuple[int, int, int],
        latent_dim: int = 8,
        channels: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        **kwargs: Any,
    ) -> None:
        self.input_shape = tuple(int(value) for value in input_shape)
        super().__init__(
            input_shape=self.input_shape,
            latent_dim=latent_dim,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

    def _initialize(self) -> None:
        self._model = None

    def _build_model(self, **kwargs: Any) -> Any:
        from ..backends.conv_autoencoder import ConvAutoencoderModel

        params = dict(self.default_params)
        params.update(kwargs)
        if "input_shape" not in params:
            raise TypeError("input_shape=(C, H, W) is required")
        return ConvAutoencoderModel(**params)

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        X_val: Any = None,
        y_val: Any = None,
        finetune: bool = False,
        **kwargs: Any,
    ) -> "ConvAutoencoderAdapter":
        if self._model is None:
            runtime_params = dict(self.params)
            workers = (self._last_fit_resources or {}).get("concrete", {}).get("num_workers")
            if workers is not None:
                runtime_params["dataloader_num_workers"] = int(workers)
            self._model = self._build_model(**runtime_params)
        del kwargs
        self._model.fit(X, y, X_val=X_val, y_val=y_val, finetune=finetune)
        return self

    def predict(self, X: Any) -> Any:
        return self._require_model().predict(X)

    def encode(self, X: Any) -> Any:
        return self._require_model().encode(X)

    def decode(self, z: Any) -> Any:
        return self._require_model().decode(z)

    def reconstruct(self, X: Any) -> Any:
        return self._require_model().reconstruct(X)

    def reconstruction_metrics(
        self,
        X: Any,
        *,
        include_ssim: bool = False,
        image_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Optional[float]]:
        return self._require_model().reconstruction_metrics(
            X, include_ssim=include_ssim, image_shape=image_shape
        )

    def _require_model(self) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before this operation")
        return self._model

    @property
    def training_history(self) -> Optional[Iterable[Dict[str, float]]]:
        return None if self._model is None else self._model.training_history

    def save(self, path: Any) -> None:
        self._require_model().save(str(path))

    def load(self, path: Any) -> None:
        if self._model is None:
            self._model = self._build_model(**self.params)
        self._model.load(str(path))


__all__ = ["ConvAutoencoderAdapter"]
