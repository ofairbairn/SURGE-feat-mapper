"""Unsupervised variational autoencoder backend for SURGE.

This backend preserves the unsupervised reconstruction-focused behavior from
``surge.var_autoenc`` while exposing a registry-friendly implementation under
``surge.model.backends``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge

try:
    from tqdm.auto import trange

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    trange = None  # type: ignore[assignment]
    TQDM_AVAILABLE = False

_LOGVAR_MIN = -10.0
_LOGVAR_MAX = 10.0

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

    The unified SURGE workflow still calls ``fit(X, y)`` for all models. For
    unsupervised reconstruction we prioritize ``X`` even when ``y`` is present.
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

#kullback-leibler divergence
def _kl_divergence_from_gaussian_params(mu: np.ndarray, logvar: np.ndarray) -> float:
    bounded_logvar = np.clip(logvar, _LOGVAR_MIN, _LOGVAR_MAX)
    kl = -0.5 * np.mean(
        1.0 + bounded_logvar - np.square(mu) - np.exp(bounded_logvar)
    )
    return float(kl)


def reconstruction_metrics(
    X_true: Any,
    X_recon: Any,
    *,
    include_ssim: bool = False,
    image_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Optional[float]]:
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
    true_frob = float(np.linalg.norm(true_arr, ord="fro"))
    # Reporting-only diagnostic; never used by the ladder gate (rmse/mse/mae only).
    relative_frobenius_error = (
        float(np.linalg.norm(diff, ord="fro") / true_frob) if true_frob > 0 else float("nan")
    )
    metrics: Dict[str, Optional[float]] = {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "relative_frobenius_error": relative_frobenius_error,
        "ssim": None,
    }

    if include_ssim:
        if image_shape is None:
            raise ValueError("image_shape must be provided when include_ssim=True")
        try:
            from skimage.metrics import structural_similarity  # type: ignore[import-not-found]
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

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            bounded_logvar = torch.clamp(logvar, min=_LOGVAR_MIN, max=_LOGVAR_MAX)
            std = torch.exp(0.5 * bounded_logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
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
    val_recon: Optional[float] = None
    val_kl: Optional[float] = None


class UnsupervisedVAEModel:
    """Neural variational autoencoder model for unsupervised SURGE workflows."""

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        hidden_dims: Any = (128, 64),
        learning_rate: float = 1e-3,
        n_epochs: int = 150,
        batch_size: int = 128,
        beta: float = 1.0,
        max_grad_norm: Optional[float] = 1.0,
        random_state: int = 42,
        device: Optional[str] = None,
        dataloader_num_workers: int = 0,
        verbose: bool = False,
        log_file: str | None = None,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for UnsupervisedVAEModel. Install torch first.")
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
        if max_grad_norm is not None and float(max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be positive or None")
        self.max_grad_norm = (
            None if max_grad_norm is None else float(max_grad_norm)
        )
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.log_file = log_file
        self.dataloader_num_workers = int(dataloader_num_workers)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[_TorchVAE] = None
        self.input_dim: Optional[int] = None
        self.target_head: Optional[Ridge] = None
        self.training_history: list[Dict[str, float]] = []
        self.is_fitted = False

    def _loss_components(
        self,
        recon_x,
        x,
        mu,
        logvar,
    ):
        bounded_logvar = torch.clamp(logvar, min=_LOGVAR_MIN, max=_LOGVAR_MAX)
        recon_loss = torch.mean((recon_x - x) ** 2)
        kl = -0.5 * torch.mean(
            1.0 + bounded_logvar - mu.pow(2) - bounded_logvar.exp()
        )
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

    def _make_loader(self, X: np.ndarray, *, shuffle: bool):
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
    ) -> "UnsupervisedVAEModel":
        X_train = _as_2d_array(X)
        _ = y_val
        target_train = _reconstruction_target(X_train, y)
        force_rebuild = not finetune
        self._build_if_needed(target_train.shape[1], force_rebuild=force_rebuild)

        if self.model is None:
            raise ValueError("Internal unsupervised VAE model was not initialized")

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
        training_start = time.perf_counter()
        epoch_iterator = (
            trange(
                1,
                self.n_epochs + 1,
                desc="Unsupervised VAE Training",
                unit="epoch",
                disable=not self.verbose,
            )
            if self.verbose and TQDM_AVAILABLE
            else range(1, self.n_epochs + 1)
        )
        for epoch in epoch_iterator:
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_samples = 0

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                recon, mu, logvar = self.model(batch_x)
                loss, recon_loss, kl = self._loss_components(recon, batch_x, mu, logvar)
                if not all(
                    torch.isfinite(value).all()
                    for value in (recon, mu, logvar, loss, recon_loss, kl)
                ):
                    raise FloatingPointError(
                        "Unsupervised VAE produced a non-finite training value "
                        f"at epoch={epoch}, batch_size={batch_x.size(0)}. "
                        f"logvar is clamped to [{_LOGVAR_MIN}, {_LOGVAR_MAX}]; "
                        "check input scale, beta, and learning_rate."
                    )
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in self.model.parameters()
                ):
                    raise FloatingPointError(
                        "Unsupervised VAE produced a non-finite gradient "
                        f"at epoch={epoch}, batch_size={batch_x.size(0)}. "
                        f"Gradient clipping was applied with max_grad_norm={self.max_grad_norm}; "
                        "check input scale, beta, and learning_rate."
                    )
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
                val_recon_total = 0.0
                val_kl_total = 0.0
                val_count = 0
                with torch.no_grad():
                    for (batch_val,) in val_loader:
                        batch_val = batch_val.to(self.device, non_blocking=True)
                        recon, mu, logvar = self.model(batch_val)
                        loss, recon_loss, kl = self._loss_components(recon, batch_val, mu, logvar)
                        if not all(
                            torch.isfinite(value).all()
                            for value in (recon, mu, logvar, loss, recon_loss, kl)
                        ):
                            raise FloatingPointError(
                                "Unsupervised VAE produced a non-finite validation value "
                                f"at epoch={epoch}, batch_size={batch_val.size(0)}."
                            )
                        bs = batch_val.size(0)
                        val_count += bs
                        val_loss_total += float(loss.detach().cpu().item()) * bs
                        val_recon_total += float(recon_loss.detach().cpu().item()) * bs
                        val_kl_total += float(kl.detach().cpu().item()) * bs
                train_state.val_loss = val_loss_total / max(1, val_count)
                train_state.val_recon = val_recon_total / max(1, val_count)
                train_state.val_kl = val_kl_total / max(1, val_count)
                self.model.train()

            row: Dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": float(train_state.train_loss),
                "train_recon": float(train_state.train_recon),
                "train_kl": float(train_state.train_kl),
                "epoch_seconds": float(time.perf_counter() - epoch_start),
                "elapsed_seconds": float(time.perf_counter() - training_start),
            }
            if train_state.val_loss is not None:
                row["val_loss"] = float(train_state.val_loss)
            if train_state.val_recon is not None:
                row["val_recon"] = float(train_state.val_recon)
            if train_state.val_kl is not None:
                row["val_kl"] = float(train_state.val_kl)
            self.training_history.append(row)

            if self.verbose and TQDM_AVAILABLE and hasattr(epoch_iterator, "set_postfix"):
                progress_metrics = {
                    "train_loss": f"{train_state.train_loss:.6f}",
                    "recon": f"{train_state.train_recon:.6f}",
                    "kl": f"{train_state.train_kl:.6f}",
                    "elapsed": f"{row['elapsed_seconds']:.1f}s",
                }
                if train_state.val_loss is not None:
                    progress_metrics["val_loss"] = f"{train_state.val_loss:.6f}"
                epoch_iterator.set_postfix(progress_metrics)
            elif self.verbose and (
                epoch == 1 or epoch % 10 == 0 or epoch == self.n_epochs
            ):
                val_text = (
                    f" val_loss={train_state.val_loss:.6f}"
                    if train_state.val_loss is not None
                    else ""
                )
                print(
                    f"[UnsupervisedVAE] epoch={epoch:03d} "
                    f"train_loss={train_state.train_loss:.6f}{val_text} "
                    f"elapsed={row['elapsed_seconds']:.1f}s"
                )

        if y is not None:
            y_arr = _as_2d_targets(y)
            if not _is_same_shape(target_train, y_arr):
                z_train = self.encode(target_train, sample=False)
                head = Ridge(alpha=1.0, random_state=self.random_state)
                head.fit(z_train, y_arr)
                self.target_head = head

        self.is_fitted = True
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
        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                tensor = torch.tensor(arr, dtype=torch.float32).to(self.device)
                mu, logvar = self.model.encode(tensor)
            metrics["kl"] = _kl_divergence_from_gaussian_params(
                mu.detach().cpu().numpy(),
                logvar.detach().cpu().numpy(),
            )
        return metrics

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(
            {
                "config": {
                    "latent_dim": self.latent_dim,
                    "hidden_dims": list(self.hidden_dims),
                    "learning_rate": self.learning_rate,
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "beta": self.beta,
                    "max_grad_norm": self.max_grad_norm,
                    "random_state": self.random_state,
                    "input_dim": self.input_dim,
                },
                "model_state": self.model.state_dict() if self.model is not None else None,
                "target_head": self.target_head,
                "training_history": list(self.training_history),
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path: str) -> None:
        import joblib

        payload = joblib.load(path)
        cfg = payload["config"]
        self.latent_dim = int(cfg["latent_dim"])
        self.hidden_dims = tuple(int(h) for h in cfg["hidden_dims"])
        self.learning_rate = float(cfg["learning_rate"])
        self.n_epochs = int(cfg["n_epochs"])
        self.batch_size = int(cfg["batch_size"])
        self.beta = float(cfg["beta"])
        saved_max_grad_norm = cfg.get("max_grad_norm", self.max_grad_norm)
        self.max_grad_norm = (
            None
            if saved_max_grad_norm is None
            else float(saved_max_grad_norm)
        )
        self.random_state = int(cfg["random_state"])
        self.input_dim = cfg.get("input_dim")
        if self.input_dim is not None:
            self._build_if_needed(int(self.input_dim), force_rebuild=True)
        if self.model is not None and payload.get("model_state") is not None:
            self.model.load_state_dict(payload["model_state"])
            self.model.to(self.device)
            self.model.eval()
        self.target_head = payload.get("target_head")
        self.training_history = list(payload.get("training_history", []))
        self.is_fitted = bool(payload.get("is_fitted", False))


__all__ = [
    "UnsupervisedVAEModel",
    "TORCH_AVAILABLE",
    "TQDM_AVAILABLE",
    "reconstruction_metrics",
]
