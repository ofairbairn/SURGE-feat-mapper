"""Deterministic autoencoder backend for SURGE unsupervised workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from .owen_vae import reconstruction_metrics

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


def _as_2d_targets(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected target array to be 1D or 2D, got shape={arr.shape}")
    return arr


def _is_same_shape(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(a.ndim == 2 and b.ndim == 2 and a.shape == b.shape)


if TORCH_AVAILABLE:

    class _TorchAE(nn.Module):
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

            last_hidden = hidden_dims[-1] if hidden_dims else input_dim
            encoder_layers.append(nn.Linear(last_hidden, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_dims = [latent_dim, *reversed(hidden_dims), input_dim]
            decoder_layers = []
            for in_dim, out_dim in zip(decoder_dims[:-2], decoder_dims[1:-1]):
                decoder_layers.append(nn.Linear(in_dim, out_dim))
                decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(decoder_dims[-2], decoder_dims[-1]))
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            z = self.encode(x)
            recon = self.decode(z)
            return recon


@dataclass
class _TorchAETrainState:
    train_loss: float
    val_loss: Optional[float] = None


class AutoencoderModel:
    """Neural deterministic autoencoder for unsupervised SURGE workflows."""

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        hidden_dims: Any = (128, 64),
        learning_rate: float = 1e-3,
        n_epochs: int = 150,
        batch_size: int = 128,
        random_state: int = 42,
        device: Optional[str] = None,
        dataloader_num_workers: int = 0,
        verbose: bool = False,
        **_: Any,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AutoencoderModel. Install torch first.")

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
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.dataloader_num_workers = int(dataloader_num_workers)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[_TorchAE] = None
        self.input_dim: Optional[int] = None
        self.target_head: Optional[Ridge] = None
        self.training_history: list[Dict[str, float]] = []
        self.is_fitted = False

    def _build_if_needed(self, input_dim: int, *, force_rebuild: bool = False) -> None:
        if self.model is not None and not force_rebuild:
            return
        self.input_dim = int(input_dim)
        torch.manual_seed(self.random_state)
        self.model = _TorchAE(
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
    ) -> "AutoencoderModel":
        X_train = _as_2d_array(X)
        _ = y_val
        target_train = X_train

        force_rebuild = not finetune
        self._build_if_needed(target_train.shape[1], force_rebuild=force_rebuild)
        if self.model is None:
            raise ValueError("Internal autoencoder model was not initialized")

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
                desc="Autoencoder Training",
                unit="epoch",
                disable=not self.verbose,
            )
            if self.verbose and TQDM_AVAILABLE
            else range(1, self.n_epochs + 1)
        )
        for epoch in epoch_iterator:
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            n_samples = 0

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = torch.mean((recon - batch_x) ** 2)
                loss.backward()
                optimizer.step()

                bs = batch_x.size(0)
                n_samples += bs
                epoch_loss += float(loss.detach().cpu().item()) * bs

            train_state = _TorchAETrainState(train_loss=epoch_loss / max(1, n_samples))

            if val_loader is not None:
                self.model.eval()
                val_loss_total = 0.0
                val_count = 0
                with torch.no_grad():
                    for (batch_val,) in val_loader:
                        batch_val = batch_val.to(self.device, non_blocking=True)
                        recon = self.model(batch_val)
                        loss = torch.mean((recon - batch_val) ** 2)
                        bs = batch_val.size(0)
                        val_count += bs
                        val_loss_total += float(loss.detach().cpu().item()) * bs
                train_state.val_loss = val_loss_total / max(1, val_count)
                self.model.train()

            row: Dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": float(train_state.train_loss),
                "epoch_seconds": float(time.perf_counter() - epoch_start),
                "elapsed_seconds": float(time.perf_counter() - training_start),
            }
            if train_state.val_loss is not None:
                row["val_loss"] = float(train_state.val_loss)
            self.training_history.append(row)

            if self.verbose and TQDM_AVAILABLE and hasattr(epoch_iterator, "set_postfix"):
                progress_metrics = {
                    "train_loss": f"{train_state.train_loss:.6f}",
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
                    f"[Autoencoder] epoch={epoch:03d} "
                    f"train_loss={train_state.train_loss:.6f}{val_text} "
                    f"elapsed={row['elapsed_seconds']:.1f}s"
                )

        if y is not None:
            y_arr = _as_2d_targets(y)
            if not _is_same_shape(target_train, y_arr):
                z_train = self.encode(target_train)
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
                z = self.model.encode(batch_x)
                recon = self.model.decode(z)
                recons.append(recon.detach().cpu().numpy())
        return np.vstack(recons).astype(np.float32)

    def predict(self, X: Any) -> np.ndarray:
        arr = _as_2d_array(X)
        if self.target_head is not None:
            z = self.encode(arr)
            pred = np.asarray(self.target_head.predict(z), dtype=np.float32)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            return pred
        return self._predict_recon(arr)

    def encode(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before encoding")
        arr = _as_2d_array(X)
        self.model.eval()
        loader = self._make_loader(arr, shuffle=False)
        latents = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                z = self.model.encode(batch_x)
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

    def reconstruct(self, X: Any) -> np.ndarray:
        return self.predict(X)

    def reconstruction_metrics(
        self,
        X: Any,
        *,
        include_ssim: bool = False,
        image_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Optional[float]]:
        arr = _as_2d_array(X)
        recon = self.reconstruct(arr)
        metrics = reconstruction_metrics(
            arr,
            recon,
            include_ssim=include_ssim,
            image_shape=image_shape,
        )
        latent = self.encode(arr)
        metrics["latent_var_mean"] = float(np.var(latent, axis=0).mean())
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


__all__ = ["AutoencoderModel", "TORCH_AVAILABLE", "TQDM_AVAILABLE"]
