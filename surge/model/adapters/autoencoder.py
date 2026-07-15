"""Adapter registering ``pytorch.autoencoder``."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from ...hpc import ResourceProfile
from ..base import BaseModelAdapter, ModelInfo

_AUTOENCODER_INFO = ModelInfo(
    architecture=(
        "Deterministic MLP autoencoder with an encoder bottleneck and decoder "
        "trained using reconstruction MSE."
    ),
    use_cases=[
        "Reconstruction-first representation learning",
        "Anomaly scoring from reconstruction error",
        "Latent embedding extraction for downstream analysis",
    ],
    not_for=[
        "Uncertainty-aware latent sampling",
        "Probabilistic latent interpolation requiring posterior variance",
    ],
    strengths=[
        "Typically stronger reconstruction fidelity than VAE under equal capacity",
        "Simpler optimization objective and tuning",
    ],
    weaknesses=[
        "No explicit probabilistic latent structure",
        "Latent manifold may be less smooth for interpolation than VAE",
    ],
    references=[
        "Hinton & Salakhutdinov (2006) Reducing the Dimensionality of Data with Neural Networks.",
    ],
)

_AUTOENCODER_PROFILE = ResourceProfile(
    name="pytorch.autoencoder",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="dataloader_workers",
    notes="num_workers maps to torch DataLoader workers.",
)


class AutoencoderAdapter(BaseModelAdapter):
    """Deterministic autoencoder adapter for SURGE."""

    name = "pytorch.autoencoder"
    backend = "pytorch"
    task_type = "unsupervised"
    uses_internal_preprocessing = False
    handles_output_scaling = False
    resource_profile = _AUTOENCODER_PROFILE
    _INFO = _AUTOENCODER_INFO
    default_params: Dict[str, Any] = {
        "latent_dim": 8,
        "hidden_dims": (128, 64),
        "learning_rate": 1e-3,
        "n_epochs": 150,
        "batch_size": 128,
        "random_state": 42,
        "verbose": False,
    }

    def _initialize(self) -> None:
        self._model = None

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib

        params = dict(self.default_params)
        params.update(kwargs)
        if "hidden_dims" not in params and "hidden_dim" in params:
            params["hidden_dims"] = params.pop("hidden_dim")
        mod = importlib.import_module("surge.model.backends.autoencoder")
        return mod.AutoencoderModel(**params)

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        X_val: Any = None,
        y_val: Any = None,
        finetune: bool = False,
        **kwargs: Any,
    ) -> "AutoencoderAdapter":
        if self._model is None and not finetune:
            runtime_params = dict(self.params)
            res = self._last_fit_resources or {}
            workers = res.get("concrete", {}).get("num_workers")
            if workers is not None:
                runtime_params["dataloader_num_workers"] = int(workers)
            self._model = self._build_model(**runtime_params)
        if self._model is None:
            self._model = self._build_model(**self.params)
        _ = kwargs
        self._model.fit(X, y, X_val=X_val, y_val=y_val, finetune=finetune)
        return self

    def predict(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X)

    def encode(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before encoding")
        return self._model.encode(X)

    def decode(self, z: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before decoding")
        return self._model.decode(z)

    def reconstruct(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before reconstructing")
        return self._model.reconstruct(X)

    def reconstruction_metrics(
        self,
        X: Any,
        *,
        include_ssim: bool = False,
        image_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Optional[float]]:
        if self._model is None:
            raise ValueError("Model must be fitted before computing reconstruction metrics")
        return self._model.reconstruction_metrics(
            X,
            include_ssim=include_ssim,
            image_shape=image_shape,
        )

    @property
    def training_history(self) -> Optional[Iterable[Dict[str, float]]]:
        if self._model is None:
            return None
        return getattr(self._model, "training_history", None)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        if self._model is None:
            self._model = self._build_model(**self.params)
        self._model.load(str(path))
