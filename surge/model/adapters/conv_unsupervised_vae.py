"""Adapter registering ``pytorch.conv_unsupervised_vae``."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...hpc import ResourceProfile
from ..base import ModelInfo
from .conv_autoencoder import ConvAutoencoderAdapter


class ConvUnsupervisedVAEAdapter(ConvAutoencoderAdapter):
    """SURGE adapter for an unsupervised convolutional beta-VAE."""

    name = "pytorch.conv_unsupervised_vae"
    resource_profile = ResourceProfile(
        name=name,
        supports_cpu=True,
        supports_gpu=True,
        worker_semantics="dataloader_workers",
        notes="num_workers maps to torch DataLoader workers.",
    )
    _INFO = ModelInfo(
        architecture=(
            "Conv2d encoder producing Gaussian latent parameters, a sampled "
            "vector bottleneck, and a ConvTranspose2d decoder."
        ),
        use_cases=[
            "Spatial latent representation learning",
            "Image reconstruction and anomaly detection",
            "Sampling and latent-space visualization",
        ],
        not_for=["Non-grid data without a meaningful image reshape"],
        strengths=[
            "Retains spatial inductive bias",
            "Supports deterministic mean embeddings and sampled embeddings",
        ],
        weaknesses=[
            "KL regularization can trade reconstruction fidelity for latent smoothness"
        ],
        references=[
            "Kingma & Welling (2014) 'Auto-Encoding Variational Bayes' ICLR 2014."
        ],
    )
    default_params: Dict[str, Any] = {
        **ConvAutoencoderAdapter.default_params,
        "beta": 1.0,
        "max_grad_norm": 1.0,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        from ..backends.conv_unsupervised_vae import ConvUnsupervisedVAEModel

        params = dict(self.default_params)
        params.update(kwargs)
        if "input_shape" not in params:
            raise TypeError("input_shape=(C, H, W) is required")
        return ConvUnsupervisedVAEModel(**params)

    def encode(self, X: Any, *, sample: bool = False) -> np.ndarray:
        return self._require_model().encode(X, sample=sample)

    def reconstruct(self, X: Any, *, sample: bool = False) -> np.ndarray:
        return self._require_model().reconstruct(X, sample=sample)


__all__ = ["ConvUnsupervisedVAEAdapter"]
