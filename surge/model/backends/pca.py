"""Principal component analysis backend for SURGE unsupervised workflows."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA

try:
	from tqdm.auto import tqdm

	TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
	tqdm = None  # type: ignore[assignment]
	TQDM_AVAILABLE = False


def _as_2d_array(X: Any) -> np.ndarray:
	arr = np.asarray(X, dtype=np.float64)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 1)
	if arr.ndim != 2:
		raise ValueError(f"Expected a 2D array, got shape={arr.shape}")
	return arr


class PCAModel:
	"""PCA reconstruction model with encoder/decoder semantics."""

	def __init__(
		self,
		*,
		n_components: Optional[int] = 8,
		max_components: int = 32,
		whiten: bool = False,
		svd_solver: str = "auto",
		random_state: Optional[int] = 42,
		verbose: bool = False,
		**_: Any,
	) -> None:
		if n_components is not None and int(n_components) < 1:
			raise ValueError("n_components must be at least 1 or None")
		if int(max_components) < 1:
			raise ValueError("max_components must be at least 1")
		self.n_components = None if n_components is None else int(n_components)
		self.max_components = int(max_components)
		self.whiten = bool(whiten)
		self.svd_solver = str(svd_solver)
		self.random_state = random_state
		self.verbose = bool(verbose)
		self.model: Optional[PCA] = None
		self.n_components_: Optional[int] = None
		self.is_fitted = False
		self.training_history: list[Dict[str, float]] = []

	def fit(self, X: Any, y: Any = None, **_: Any) -> "PCAModel":
		arr = _as_2d_array(X)
		_ = y
		upper = min(arr.shape[0], arr.shape[1], self.max_components)
		if upper < 1:
			raise ValueError(f"PCA requires at least one sample and feature, got {arr.shape}")
		requested = upper if self.n_components is None else self.n_components
		effective_components = min(requested, upper)
		self.model = PCA(
			n_components=effective_components,
			whiten=self.whiten,
			svd_solver=self.svd_solver,
			random_state=self.random_state,
		)
		fit_start = time.perf_counter()
		if self.verbose and TQDM_AVAILABLE:
			with tqdm(total=1, desc="PCA Fitting", unit="fit") as progress:
				self.model.fit(arr)
				progress.update(1)
				progress.set_postfix(
					components=effective_components,
					elapsed=f"{time.perf_counter() - fit_start:.2f}s",
				)
		else:
			if self.verbose:
				print("[PCA] fitting...", flush=True)
			self.model.fit(arr)
			if self.verbose:
				print(
					f"[PCA] fit complete components={effective_components} "
					f"elapsed={time.perf_counter() - fit_start:.2f}s",
					flush=True,
				)
		self.n_components_ = int(self.model.n_components_)
		self.is_fitted = True
		return self

	def _require_model(self) -> PCA:
		if self.model is None or not self.is_fitted:
			raise ValueError("Model must be fitted before use")
		return self.model

	def encode(self, X: Any) -> np.ndarray:
		return np.asarray(self._require_model().transform(_as_2d_array(X)), dtype=np.float64)

	def decode(self, z: Any) -> np.ndarray:
		return np.asarray(self._require_model().inverse_transform(_as_2d_array(z)), dtype=np.float64)

	def reconstruct(self, X: Any) -> np.ndarray:
		return self.decode(self.encode(X))

	def predict(self, X: Any) -> np.ndarray:
		return self.reconstruct(X)

	def reconstruction_metrics(self, X: Any) -> Dict[str, float]:
		arr = _as_2d_array(X)
		latent = self.encode(arr)
		diff = arr - self.decode(latent)
		mse = float(np.mean(np.square(diff)))
		true_frob = float(np.linalg.norm(arr, ord="fro"))
		# Reporting-only diagnostic; never used by the ladder gate (rmse/mse/mae only).
		relative_frobenius_error = (
			float(np.linalg.norm(diff, ord="fro") / true_frob) if true_frob > 0 else float("nan")
		)
		return {
			"mse": mse,
			"mae": float(np.mean(np.abs(diff))),
			"rmse": float(np.sqrt(mse)),
			"relative_frobenius_error": relative_frobenius_error,
			"latent_var_mean": float(np.var(latent, axis=0).mean()),
			"explained_variance_ratio_sum": float(
				np.sum(self._require_model().explained_variance_ratio_)
			),
		}

	def save(self, path: str) -> None:
		joblib.dump(
			{
				"config": {
					"n_components": self.n_components,
					"max_components": self.max_components,
					"whiten": self.whiten,
					"svd_solver": self.svd_solver,
					"random_state": self.random_state,
					"verbose": self.verbose,
				},
				"model": self.model,
				"n_components_": self.n_components_,
				"is_fitted": self.is_fitted,
			},
			path,
		)

	def load(self, path: str) -> None:
		payload = joblib.load(path)
		config = payload["config"]
		self.n_components = config["n_components"]
		self.max_components = int(config["max_components"])
		self.whiten = bool(config["whiten"])
		self.svd_solver = str(config["svd_solver"])
		self.random_state = config["random_state"]
		self.verbose = bool(config.get("verbose", self.verbose))
		self.model = payload["model"]
		self.n_components_ = payload.get("n_components_")
		self.is_fitted = bool(payload.get("is_fitted", self.model is not None))


__all__ = ["PCAModel", "TQDM_AVAILABLE"]
