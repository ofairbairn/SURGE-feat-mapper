"""Variational autoencoder adapters for SURGE.

This module provides two SURGE-compatible adapters:

1) SklearnVariationalAutoencoderAdapter
   A lightweight, linear variational bottleneck built on top of PCA.
   It is useful when you need a dependency-light baseline that follows the
   same adapter contract as other SURGE models.

2) PyTorchVariationalAutoencoderAdapter
   A neural VAE trained with reconstruction + KL divergence loss.
   This is the primary implementation for expressive latent embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state

from .hpc import ResourceProfile
from .model.base import BaseModelAdapter

try:
	import torch
	import torch.nn as nn
	import torch.optim as optim
	from torch.utils.data import DataLoader, TensorDataset

	TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
	TORCH_AVAILABLE = False
	torch = None
	nn = None
	optim = None
	DataLoader = None
	TensorDataset = None


def _as_2d_array(X: Any) -> np.ndarray:
	arr = np.asarray(X, dtype=np.float32)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 1)
	if arr.ndim != 2:
		raise ValueError(f"Expected a 2D array, got shape={arr.shape}")
	return arr


def _reconstruction_target(X: np.ndarray, y: Any) -> np.ndarray:
	"""Resolve unsupervised reconstruction target.

	The SURGE engine currently calls fit(X, y) for all models. For unsupervised
	reconstruction we prioritize X. If a user passes y with the same shape as X,
	we still default to X to preserve the unsupervised contract.
	"""
	_ = y
	return X


def _as_2d_targets(y: Any) -> np.ndarray:
	arr = np.asarray(y, dtype=np.float32)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 1)
	if arr.ndim != 2:
		raise ValueError(f"Expected target array to be 1D or 2D, got shape={arr.shape}")
	return arr


def _is_same_shape(a: np.ndarray, b: np.ndarray) -> bool:
	return bool(a.ndim == 2 and b.ndim == 2 and a.shape == b.shape)


def _kl_divergence_from_gaussian_params(mu: np.ndarray, logvar: np.ndarray) -> float:
	kl = -0.5 * np.mean(1.0 + logvar - np.square(mu) - np.exp(logvar))
	return float(kl)


def reconstruction_metrics(
	X_true: Any,
	X_recon: Any,
	*,
	include_ssim: bool = False,
	image_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Optional[float]]:
	"""Compute reconstruction quality metrics.

	SSIM is optional and only computed when scikit-image is installed and
	image_shape is provided.
	"""
	true_arr = _as_2d_array(X_true)
	recon_arr = _as_2d_array(X_recon)
	if true_arr.shape != recon_arr.shape:
		raise ValueError(
			f"Shape mismatch between true and reconstructed arrays: "
			f"{true_arr.shape} vs {recon_arr.shape}"
		)

	diff = true_arr - recon_arr
	mse = float(np.mean(np.square(diff)))
	mae = float(np.mean(np.abs(diff)))
	metrics: Dict[str, Optional[float]] = {
		"mse": mse,
		"mae": mae,
		"rmse": float(np.sqrt(mse)),
		"ssim": None,
	}

	if include_ssim:
		if image_shape is None:
			raise ValueError("image_shape must be provided when include_ssim=True")
		try:
			from skimage.metrics import structural_similarity
		except ImportError:
			metrics["ssim"] = None
			return metrics

		flat_size = int(np.prod(image_shape))
		if true_arr.shape[1] != flat_size:
			raise ValueError(
				"image_shape does not match flattened feature dimension: "
				f"{image_shape} -> {flat_size}, feature_dim={true_arr.shape[1]}"
			)

		ssim_scores = []
		for idx in range(true_arr.shape[0]):
			a = true_arr[idx].reshape(image_shape)
			b = recon_arr[idx].reshape(image_shape)
			data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
			if data_range <= 0.0:
				data_range = 1.0
			score = structural_similarity(a, b, data_range=data_range)
			ssim_scores.append(float(score))
		metrics["ssim"] = float(np.mean(ssim_scores))
	return metrics


_SKLEARN_VAE_PROFILE = ResourceProfile(
	name="sklearn.variational_autoencoder",
	supports_cpu=True,
	supports_gpu=False,
	worker_semantics="none",
	notes="Linear VAE-style adapter (PCA latent bottleneck).",
)
##################### SKLEARN VAE ADAPTER #######################

class SklearnVariationalAutoencoderAdapter(BaseModelAdapter):
	"""Dependency-light VAE-style adapter using a linear latent bottleneck.

	This implementation uses PCA to learn an encoder/decoder pair and models a
	Gaussian latent posterior with per-latent-dimension variance estimated from
	latent codes. It is intentionally simple and fast, suitable as a baseline.
	"""

	name = "sklearn.variational_autoencoder"
	backend = "sklearn"
	uses_internal_preprocessing = False
	handles_output_scaling = False
	supports_sklearn_interface = True
	resource_profile = _SKLEARN_VAE_PROFILE

	def _build_model(self, **kwargs: Any) -> Dict[str, Any]:
		latent_dim = int(kwargs.get("latent_dim", 8))
		random_state = kwargs.get("random_state", 42)
		svd_solver = kwargs.get("svd_solver", "auto")
		return {
			"latent_dim": latent_dim,
			"random_state": random_state,
			"svd_solver": svd_solver,
			"pca": None,
			"latent_logvar": None,
			"target_head": None,
			"is_fitted": False,
		}

	@staticmethod
	def _resolve_pca_hyperparams(
		requested_latent_dim: int,
		svd_solver: str,
		n_samples: int,
		n_features: int,
	) -> Tuple[int, str]:
		max_components = int(min(n_samples, n_features))
		if max_components < 1:
			raise ValueError("PCA requires at least one sample and one feature")

		solver = str(svd_solver)
		# sklearn PCA with arpack requires n_components < min(n_samples, n_features).
		if solver == "arpack":
			upper = max_components - 1
			if upper < 1:
				solver = "full"
				upper = max_components
		else:
			upper = max_components

		resolved_dim = max(1, min(int(requested_latent_dim), int(upper)))
		return resolved_dim, solver

	def fit(self, X: Any, y: Any = None) -> "SklearnVariationalAutoencoderAdapter":
		arr = _as_2d_array(X)
		target = _reconstruction_target(arr, y)
		requested_dim = int(self._model["latent_dim"])
		solver = str(self._model.get("svd_solver", "auto"))
		resolved_dim, resolved_solver = self._resolve_pca_hyperparams(
			requested_latent_dim=requested_dim,
			svd_solver=solver,
			n_samples=target.shape[0],
			n_features=target.shape[1],
		)
		self._model["resolved_latent_dim"] = resolved_dim
		self._model["resolved_svd_solver"] = resolved_solver
		pca = PCA(
			n_components=resolved_dim,
			svd_solver=resolved_solver,
			random_state=self._model["random_state"],
		)
		self._model["pca"] = pca

		z_mean = pca.fit_transform(target)

		# Optional supervised compatibility head: if y is provided and does not
		# match reconstruction dimensionality, learn latent -> y mapping so
		# predict() integrates with SURGE's regression metric path.
		target_head = None
		if y is not None:
			y_arr = _as_2d_targets(y)
			if not _is_same_shape(target, y_arr):
				target_head = Ridge(alpha=1.0, random_state=self._model["random_state"])
				target_head.fit(z_mean, y_arr)
		self._model["target_head"] = target_head

		latent_var = np.var(z_mean, axis=0, ddof=1)
		latent_var = np.maximum(latent_var, 1e-6)
		self._model["latent_logvar"] = np.log(latent_var)
		self._model["is_fitted"] = True
		return self

	def _ensure_fitted(self) -> None:
		if self._model is None or not bool(self._model.get("is_fitted", False)):
			raise ValueError("Model must be fitted before calling this method")

	def encode(self, X: Any, *, sample: bool = False) -> np.ndarray:
		self._ensure_fitted()
		arr = _as_2d_array(X)
		pca: PCA = self._model["pca"]
		mu = pca.transform(arr)
		if not sample:
			return mu

		logvar = np.asarray(self._model["latent_logvar"], dtype=np.float32)
		rng = check_random_state(self._model["random_state"])
		eps = rng.normal(0.0, 1.0, size=mu.shape).astype(np.float32)
		return mu + eps * np.exp(0.5 * logvar)

	def decode(self, z: Any) -> np.ndarray:
		self._ensure_fitted()
		latents = _as_2d_array(z)
		pca: PCA = self._model["pca"]
		return pca.inverse_transform(latents)

	def reconstruct(self, X: Any, *, sample: bool = False) -> np.ndarray:
		z = self.encode(X, sample=sample)
		return self.decode(z)

	def predict(self, X: Any) -> np.ndarray:
		self._ensure_fitted()
		head = self._model.get("target_head")
		if head is not None:
			latent = self.encode(X, sample=False)
			pred = np.asarray(head.predict(latent), dtype=np.float32)
			if pred.ndim == 1:
				pred = pred.reshape(-1, 1)
			return pred
		return self.reconstruct(X, sample=False)

	def reconstruction_metrics(
		self,
		X: Any,
		*,
		include_ssim: bool = False,
		image_shape: Optional[Tuple[int, ...]] = None,
	) -> Dict[str, Optional[float]]:
		arr = _as_2d_array(X)
		recon = self.reconstruct(arr, sample=False)
		metrics = reconstruction_metrics(
			arr,
			recon,
			include_ssim=include_ssim,
			image_shape=image_shape,
		)

		mu = self.encode(arr, sample=False)
		logvar = np.broadcast_to(np.asarray(self._model["latent_logvar"]), mu.shape)
		metrics["kl"] = _kl_divergence_from_gaussian_params(mu, logvar)
		return metrics


_TORCH_VAE_PROFILE = ResourceProfile(
	name="pytorch.variational_autoencoder",
	supports_cpu=True,
	supports_gpu=True,
	worker_semantics="dataloader_workers",
	notes="num_workers maps to torch DataLoader workers.",
)


if TORCH_AVAILABLE:

	class _TorchVAE(nn.Module):
		def __init__(
			self,
			input_dim: int,
			latent_dim: int,
			hidden_dims: Sequence[int],
		) -> None:
			super().__init__()
			dims = [input_dim, *hidden_dims]

			encoder_layers = []
			for in_dim, out_dim in zip(dims[:-1], dims[1:]):
				encoder_layers.append(nn.Linear(in_dim, out_dim))
				encoder_layers.append(nn.ReLU())
			self.encoder = nn.Sequential(*encoder_layers)

			last_hidden = hidden_dims[-1] if hidden_dims else input_dim
			self.fc_mu = nn.Linear(last_hidden, latent_dim)
			self.fc_logvar = nn.Linear(last_hidden, latent_dim)

			decoder_dims = [latent_dim, *reversed(hidden_dims), input_dim]
			decoder_layers = []
			for in_dim, out_dim in zip(decoder_dims[:-2], decoder_dims[1:-1]):
				decoder_layers.append(nn.Linear(in_dim, out_dim))
				decoder_layers.append(nn.ReLU())
			decoder_layers.append(nn.Linear(decoder_dims[-2], decoder_dims[-1]))
			self.decoder = nn.Sequential(*decoder_layers)

		def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
			h = self.encoder(x)
			return self.fc_mu(h), self.fc_logvar(h)

		def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			return mu + eps * std

		def decode(self, z: torch.Tensor) -> torch.Tensor:
			return self.decoder(z)

		def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
			mu, logvar = self.encode(x)
			z = self.reparameterize(mu, logvar)
			recon = self.decode(z)
			return recon, mu, logvar


@dataclass
class _TorchVAETrainState:
	train_loss: float
	train_recon: float
	train_kl: float
	val_loss: Optional[float] = None

##################### PYTORCH VAE ESTIMATOR #######################
class _TorchVAEEstimator:
	def __init__(
		self,
		*,
		latent_dim: int = 8,
		hidden_dims: Any = (128, 64),
		learning_rate: float = 1e-3,
		n_epochs: int = 150,
		batch_size: int = 128,
		beta: float = 1.0,
		random_state: int = 42,
		device: Optional[str] = None,
		dataloader_num_workers: int = 0,
		verbose: bool = False,
	) -> None:
		if not TORCH_AVAILABLE:
			raise ImportError("PyTorch is required for the torch VAE adapter. Install torch first.")
		self.latent_dim = int(latent_dim)
		if isinstance(hidden_dims, (int, np.integer)):
			hidden_dims = (int(hidden_dims),)
		elif isinstance(hidden_dims, list):
			hidden_dims = tuple(hidden_dims)
		if not isinstance(hidden_dims, tuple):
			raise TypeError("hidden_dims must be an int, list[int], or tuple[int, ...]")
		self.hidden_dims = tuple(int(h) for h in hidden_dims)
		self.learning_rate = float(learning_rate)
		self.n_epochs = int(n_epochs)
		self.batch_size = int(batch_size)
		self.beta = float(beta)
		self.random_state = int(random_state)
		self.verbose = bool(verbose)
		self.dataloader_num_workers = int(dataloader_num_workers)
		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device(device)

		self.model: Optional[_TorchVAE] = None
		self.input_dim: Optional[int] = None
		self.target_head: Optional[Ridge] = None
		self.training_history: list[Dict[str, float]] = []

	def _loss_components(
		self,
		recon_x: torch.Tensor,
		x: torch.Tensor,
		mu: torch.Tensor,
		logvar: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		recon_loss = torch.mean((recon_x - x) ** 2)
		kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
		total = recon_loss + self.beta * kl
		return total, recon_loss, kl

	def _build_if_needed(self, input_dim: int, *, force_rebuild: bool = False) -> None:
		if self.model is not None and not force_rebuild:
			return
		self.input_dim = int(input_dim)
		torch.manual_seed(self.random_state)
		self.model = _TorchVAE(
			input_dim=self.input_dim,
			latent_dim=self.latent_dim,
			hidden_dims=self.hidden_dims,
		).to(self.device)

	def _make_loader(self, X: np.ndarray, *, shuffle: bool) -> DataLoader:
		tensor = torch.tensor(X, dtype=torch.float32)
		dataset = TensorDataset(tensor)
		return DataLoader(
			dataset,
			batch_size=self.batch_size,
			shuffle=shuffle,
			num_workers=self.dataloader_num_workers,
			pin_memory=(self.device.type == "cuda"),
		)

	def fit(
		self,
		X: Any,
		y: Any = None,
		*,
		X_val: Any = None,
		y_val: Any = None,
		finetune: bool = False,
	) -> "_TorchVAEEstimator":
		X_train = _as_2d_array(X)
		_ = y_val
		target_train = _reconstruction_target(X_train, y)
		force_rebuild = not finetune
		self._build_if_needed(target_train.shape[1], force_rebuild=force_rebuild)

		if self.model is None:
			raise ValueError("Internal torch VAE model was not initialized")

		optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
		train_loader = self._make_loader(target_train, shuffle=True)
		val_loader = (
			self._make_loader(_as_2d_array(X_val), shuffle=False)
			if X_val is not None
			else None
		)

		self.training_history.clear()
		self.target_head = None
		self.model.train()
		for epoch in range(1, self.n_epochs + 1):
			epoch_loss = 0.0
			epoch_recon = 0.0
			epoch_kl = 0.0
			n_samples = 0

			for (batch_x,) in train_loader:
				batch_x = batch_x.to(self.device, non_blocking=True)
				optimizer.zero_grad()
				recon, mu, logvar = self.model(batch_x)
				loss, recon_loss, kl = self._loss_components(recon, batch_x, mu, logvar)
				loss.backward()
				optimizer.step()

				bs = batch_x.size(0)
				n_samples += bs
				epoch_loss += float(loss.detach().cpu().item()) * bs
				epoch_recon += float(recon_loss.detach().cpu().item()) * bs
				epoch_kl += float(kl.detach().cpu().item()) * bs

			train_state = _TorchVAETrainState(
				train_loss=epoch_loss / max(1, n_samples),
				train_recon=epoch_recon / max(1, n_samples),
				train_kl=epoch_kl / max(1, n_samples),
			)

			if val_loader is not None:
				self.model.eval()
				val_loss_total = 0.0
				val_count = 0
				with torch.no_grad():
					for (batch_val,) in val_loader:
						batch_val = batch_val.to(self.device, non_blocking=True)
						recon, mu, logvar = self.model(batch_val)
						loss, _, _ = self._loss_components(recon, batch_val, mu, logvar)
						bs = batch_val.size(0)
						val_count += bs
						val_loss_total += float(loss.detach().cpu().item()) * bs
				train_state.val_loss = val_loss_total / max(1, val_count)
				self.model.train()

			row: Dict[str, float] = {
				"epoch": float(epoch),
				"train_loss": float(train_state.train_loss),
				"train_recon": float(train_state.train_recon),
				"train_kl": float(train_state.train_kl),
			}
			if train_state.val_loss is not None:
				row["val_loss"] = float(train_state.val_loss)
			self.training_history.append(row)

			if self.verbose and (epoch == 1 or epoch % 10 == 0 or epoch == self.n_epochs):
				if train_state.val_loss is not None:
					print(
						"[TorchVAE] "
						f"epoch={epoch:03d} "
						f"train_loss={train_state.train_loss:.6f} "
						f"val_loss={train_state.val_loss:.6f}"
					)
				else:
					print(
						"[TorchVAE] "
						f"epoch={epoch:03d} "
						f"train_loss={train_state.train_loss:.6f}"
					)

		if y is not None:
			y_arr = _as_2d_targets(y)
			if not _is_same_shape(target_train, y_arr):
				z_train = self.encode(target_train, sample=False)
				head = Ridge(alpha=1.0, random_state=self.random_state)
				head.fit(z_train, y_arr)
				self.target_head = head
		return self

	def _predict_recon(self, X: np.ndarray) -> np.ndarray:
		if self.model is None:
			raise ValueError("Model must be fitted before predicting")
		self.model.eval()
		loader = self._make_loader(X, shuffle=False)
		recons = []
		with torch.no_grad():
			for (batch_x,) in loader:
				batch_x = batch_x.to(self.device, non_blocking=True)
				mu, _ = self.model.encode(batch_x)
				recon = self.model.decode(mu)
				recons.append(recon.detach().cpu().numpy())
		return np.vstack(recons).astype(np.float32)

	def predict(self, X: Any) -> np.ndarray:
		arr = _as_2d_array(X)
		if self.target_head is not None:
			z = self.encode(arr, sample=False)
			pred = np.asarray(self.target_head.predict(z), dtype=np.float32)
			if pred.ndim == 1:
				pred = pred.reshape(-1, 1)
			return pred
		return self._predict_recon(arr)

	def encode(self, X: Any, *, sample: bool = False) -> np.ndarray:
		if self.model is None:
			raise ValueError("Model must be fitted before encoding")
		arr = _as_2d_array(X)
		self.model.eval()
		loader = self._make_loader(arr, shuffle=False)
		latents = []
		with torch.no_grad():
			for (batch_x,) in loader:
				batch_x = batch_x.to(self.device, non_blocking=True)
				mu, logvar = self.model.encode(batch_x)
				if sample:
					z = self.model.reparameterize(mu, logvar)
				else:
					z = mu
				latents.append(z.detach().cpu().numpy())
		return np.vstack(latents).astype(np.float32)

	def decode(self, z: Any) -> np.ndarray:
		if self.model is None:
			raise ValueError("Model must be fitted before decoding")
		latents = _as_2d_array(z)
		self.model.eval()
		tensor = torch.tensor(latents, dtype=torch.float32).to(self.device)
		with torch.no_grad():
			recon = self.model.decode(tensor)
		return recon.detach().cpu().numpy().astype(np.float32)

##################### PYTORCH VAE Adapter #######################
class PyTorchVariationalAutoencoderAdapter(BaseModelAdapter):
	"""Neural variational autoencoder adapter for SURGE."""

	name = "pytorch.variational_autoencoder"
	backend = "pytorch"
	uses_internal_preprocessing = False
	handles_output_scaling = False
	resource_profile = _TORCH_VAE_PROFILE

	def __init__(self, **kwargs: Any) -> None:
		if not TORCH_AVAILABLE:
			raise ImportError(
				"PyTorch VAE adapter not available. Install torch: pip install torch"
			)
		super().__init__(**kwargs)

	def _initialize(self) -> None:
		self._model = None

	def _build_model(self, **kwargs: Any) -> _TorchVAEEstimator:
		if "hidden_dims" not in kwargs and "hidden_dim" in kwargs:
			kwargs = dict(kwargs)
			kwargs["hidden_dims"] = kwargs.pop("hidden_dim")
		return _TorchVAEEstimator(**kwargs)

	def fit(
		self,
		X: Any,
		y: Any = None,
		*,
		X_val: Any = None,
		y_val: Any = None,
		finetune: bool = False,
		**kwargs: Any,
	) -> "PyTorchVariationalAutoencoderAdapter":
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
		if sample:
			return self.decode(self.encode(X, sample=True))
		return self.predict(X)

	def reconstruction_metrics(
		self,
		X: Any,
		*,
		include_ssim: bool = False,
		image_shape: Optional[Tuple[int, ...]] = None,
	) -> Dict[str, Optional[float]]:
		arr = _as_2d_array(X)
		recon = self.reconstruct(arr, sample=False)
		metrics = reconstruction_metrics(
			arr,
			recon,
			include_ssim=include_ssim,
			image_shape=image_shape,
		)
		latent = self.encode(arr, sample=False)
		metrics["latent_var_mean"] = float(np.var(latent, axis=0).mean())
		return metrics

	@property
	def training_history(self) -> Optional[Iterable[Dict[str, float]]]:
		if self._model is None:
			return None
		return getattr(self._model, "training_history", None)


__all__ = [
	"TORCH_AVAILABLE",
	"SklearnVariationalAutoencoderAdapter",
	"PyTorchVariationalAutoencoderAdapter",
	"reconstruction_metrics",
]