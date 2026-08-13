"""Adapter registering ``pytorch.unsupervised_vae``."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from ...hpc import ResourceProfile
from ..base import BaseModelAdapter, ModelInfo

_UNSUPERVISED_VAE_INFO = ModelInfo(
    architecture=(
        "Unsupervised variational autoencoder with an MLP encoder/decoder. "
        "The encoder maps inputs to Gaussian latent parameters, the decoder "
        "reconstructs the original inputs, and training minimizes "
        "reconstruction loss plus beta-weighted KL divergence."
    ),
    use_cases=[
        "Learning compact latent representations of tabular or flattened image data",
        "Reconstruction-based anomaly detection and embedding analysis",
        "Latent-space visualization with UMAP or t-SNE",
    ],
    not_for=[
        "Supervised prediction benchmarks where a dedicated regressor or classifier is the goal",
        "High-fidelity image generation workloads",
    ],
    strengths=[
        "Preserves an unsupervised reconstruction objective",
        "Exposes encode/decode/reconstruct methods for downstream analysis",
        "Works naturally with latent manifold visualization pipelines",
    ],
    weaknesses=[
        "Unified workflow support for unsupervised tasks is still lighter than regression/classification",
        "Flattened inputs may lose spatial inductive bias compared to convolutional VAEs",
    ],
    references=[
        "Kingma & Welling (2014) 'Auto-Encoding Variational Bayes' ICLR 2014.",
    ],
)

_UNSUPERVISED_VAE_PROFILE = ResourceProfile(
    name="pytorch.unsupervised_vae",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="dataloader_workers",
    notes="num_workers maps to torch DataLoader workers.",
)


class UnsupervisedVAEAdapter(BaseModelAdapter):
    """Unsupervised variational autoencoder adapter for SURGE."""

    name = "pytorch.unsupervised_vae"
    backend = "pytorch"
    task_type = "unsupervised"
    uses_internal_preprocessing = False
    handles_output_scaling = False
    resource_profile = _UNSUPERVISED_VAE_PROFILE
    _INFO = _UNSUPERVISED_VAE_INFO
    default_params: Dict[str, Any] = {
        "latent_dim": 8,
        "hidden_dims": (128, 64),
        "learning_rate": 1e-3,
        "n_epochs": 150,
        "batch_size": 128,
        "beta": 1.0,
        "max_grad_norm": 1.0,
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
        mod = importlib.import_module("surge.model.backends.unsupervised_vae")
        return mod.UnsupervisedVAEModel(**params)

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        X_val: Any = None,
        y_val: Any = None,
        finetune: bool = False,
        **kwargs: Any,
    ) -> "UnsupervisedVAEAdapter":
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

    def predict(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X)

    def encode(self, X: Any, *, sample: bool = False) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model must be fitted before encoding")
        return self._model.encode(X, sample=sample)

    def decode(self, z: Any) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model must be fitted before decoding")
        return self._model.decode(z)

    def reconstruct(self, X: Any, *, sample: bool = False) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model must be fitted before reconstructing")
        return self._model.reconstruct(X, sample=sample)

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