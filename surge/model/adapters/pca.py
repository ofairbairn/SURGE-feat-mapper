"""Adapter registering ``sklearn.pca``."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...hpc import ResourceProfile
from ..base import BaseModelAdapter, ModelInfo
from ..backends.pca import PCAModel

_PCA_INFO = ModelInfo(
	architecture="Linear principal component encoder with inverse-transform decoder.",
	use_cases=[
		"Fast linear dimensionality-reduction baseline",
		"Reconstruction benchmarking for nonlinear autoencoders",
		"Latent-space visualization and clustering",
	],
	not_for=["Nonlinear representation learning", "Probabilistic latent generation"],
	strengths=["Deterministic", "Fast to fit", "Interpretable explained variance"],
	weaknesses=["Captures linear structure only", "Sensitive to feature scaling"],
	references=["Jolliffe (2002), Principal Component Analysis."],
)

_PCA_PROFILE = ResourceProfile(
	name="sklearn.pca",
	supports_cpu=True,
	supports_gpu=False,
	worker_semantics="none",
	notes="scikit-learn PCA executes on CPU.",
)


class PCAAdapter(BaseModelAdapter):
	"""SURGE adapter for PCA reconstruction and latent encoding."""

	name = "sklearn.pca"
	backend = "sklearn"
	task_type = "unsupervised"
	uses_internal_preprocessing = False
	handles_output_scaling = False
	supports_sklearn_interface = True
	resource_profile = _PCA_PROFILE
	_INFO = _PCA_INFO
	default_params: Dict[str, Any] = {
		"n_components": 8,
		"max_components": 32,
		"whiten": False,
		"svd_solver": "auto",
		"random_state": 42,
	}

	def _build_model(self, **kwargs: Any) -> PCAModel:
		params = dict(self.default_params)
		params.update(kwargs)
		return PCAModel(**params)

	def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "PCAAdapter":
		_ = kwargs
		self._model.fit(X, y)
		return self

	def encode(self, X: Any) -> Any:
		return self._model.encode(X)

	def decode(self, z: Any) -> Any:
		return self._model.decode(z)

	def reconstruct(self, X: Any) -> Any:
		return self._model.reconstruct(X)

	def reconstruction_metrics(self, X: Any) -> Dict[str, float]:
		return self._model.reconstruction_metrics(X)

	@property
	def n_components_(self) -> Optional[int]:
		return self._model.n_components_


__all__ = ["PCAAdapter"]