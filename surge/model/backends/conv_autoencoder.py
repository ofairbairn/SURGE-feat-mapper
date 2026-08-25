"""Convolutional autoencoder backend for image-shaped SURGE inputs."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from .unsupervised_vae import reconstruction_metrics

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    torch = nn = optim = DataLoader = TensorDataset = None


ImageShape = Tuple[int, int, int]


def _as_4d_array(X: Any, shape: ImageShape) -> np.ndarray:
    """Return ``X`` as a float32 ``(N, C, H, W)`` array."""
    arr = np.asarray(X, dtype=np.float32)
    try:
        return arr.reshape(-1, *shape)
    except ValueError as exc:
        raise ValueError(
            f"Input with shape {arr.shape} cannot be reshaped to (N, {shape[0]}, "
            f"{shape[1]}, {shape[2]})"
        ) from exc


def _as_2d_array(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape={arr.shape}")
    return arr


def _as_2d_targets(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected targets with a sample axis, got shape={arr.shape}")
    return arr


if TORCH_AVAILABLE:

    class _TorchConvAE(nn.Module):
        """Conv2d encoder and ConvTranspose2d decoder with a vector bottleneck."""

        def __init__(
            self,
            input_shape: ImageShape,
            latent_dim: int,
            channels: Sequence[int],
            kernel_size: int,
            stride: int,
            padding: int,
        ) -> None:
            super().__init__()
            c, h, w = input_shape
            encoder_layers = []
            in_ch = c
            spatial_shapes = [(h, w)]
            for out_ch in channels:
                encoder_layers.extend(
                    [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding), nn.ReLU()]
                )
                in_ch = out_ch
                h = (h + 2 * padding - kernel_size) // stride + 1
                w = (w + 2 * padding - kernel_size) // stride + 1
                if h < 1 or w < 1:
                    raise ValueError(
                        "Convolution stack collapses the spatial dimensions; reduce the "
                        "number of channels/layers or adjust kernel_size, stride, and padding"
                    )
                spatial_shapes.append((h, w))

            self.encoder_conv = nn.Sequential(*encoder_layers)
            self._flat_shape = (in_ch, h, w)
            flat_dim = in_ch * h * w
            self.encoder_fc = nn.Linear(flat_dim, latent_dim)
            self.decoder_fc = nn.Linear(latent_dim, flat_dim)

            decoder_layers = []
            decoder_channels = list(reversed(channels[:-1])) + [c]
            in_ch = channels[-1]
            for layer_index, out_ch in enumerate(decoder_channels):
                current_h, current_w = spatial_shapes[-1 - layer_index]
                target_h, target_w = spatial_shapes[-2 - layer_index]
                base_h = (current_h - 1) * stride - 2 * padding + kernel_size
                base_w = (current_w - 1) * stride - 2 * padding + kernel_size
                output_padding = (target_h - base_h, target_w - base_w)
                if any(value < 0 or value >= stride for value in output_padding):
                    raise ValueError(
                        f"Cannot invert convolution at layer {layer_index}: "
                        f"output_padding={output_padding}, stride={stride}"
                    )
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_ch,
                        out_ch,
                        kernel_size,
                        stride,
                        padding,
                        output_padding=output_padding,
                    )
                )
                # Keep the requested ReLU stack, including the reconstruction layer.
                decoder_layers.append(nn.ReLU())
                in_ch = out_ch
            self.decoder_conv = nn.Sequential(*decoder_layers)

        def encode(self, x):
            encoded = self.encoder_conv(x)
            return self.encoder_fc(encoded.flatten(1))

        def decode(self, z):
            decoded = self.decoder_fc(z).view(-1, *self._flat_shape)
            return self.decoder_conv(decoded)

        def forward(self, x):
            return self.decode(self.encode(x))


class ConvAutoencoderModel:
    """Deterministic convolutional autoencoder with a 1D latent vector."""

    def __init__(
        self,
        *,
        input_shape: ImageShape,
        latent_dim: int = 8,
        channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
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
            raise ImportError("PyTorch is required for ConvAutoencoderModel. Install torch first.")
        if len(input_shape) != 3:
            raise ValueError("input_shape must be a (C, H, W) tuple")
        self.input_shape = tuple(int(value) for value in input_shape)
        if any(value <= 0 for value in self.input_shape):
            raise ValueError("Every input_shape dimension must be positive")
        if isinstance(channels, (int, np.integer)):
            channels = (int(channels),)
        self.channels = tuple(int(value) for value in channels)
        if not self.channels or any(value <= 0 for value in self.channels):
            raise ValueError("channels must contain at least one positive integer")
        self.latent_dim = int(latent_dim)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        if self.latent_dim <= 0 or self.kernel_size <= 0 or self.stride <= 0 or self.padding < 0:
            raise ValueError(
                "latent_dim, kernel_size, and stride must be positive; "
                "padding cannot be negative"
            )
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.dataloader_num_workers = int(dataloader_num_workers)
        self.verbose = bool(verbose)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[_TorchConvAE] = None
        self.target_head: Optional[Ridge] = None
        self.target_shape: Optional[Tuple[int, ...]] = None
        self.training_history: list[Dict[str, float]] = []
        self.is_fitted = False

    def _build(self) -> None:
        torch.manual_seed(self.random_state)
        self.model = _TorchConvAE(
            self.input_shape,
            self.latent_dim,
            self.channels,
            self.kernel_size,
            self.stride,
            self.padding,
        ).to(self.device)

    def _make_loader(self, X: np.ndarray, *, shuffle: bool):
        dataset = TensorDataset(torch.from_numpy(X))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        X_val: Any = None,
        y_val: Any = None,
        finetune: bool = False,
    ) -> "ConvAutoencoderModel":
        del y_val
        X_train = _as_4d_array(X, self.input_shape)
        if self.model is None or not finetune:
            self._build()
        assert self.model is not None
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_loader = self._make_loader(X_train, shuffle=True)
        val_loader = (
            self._make_loader(_as_4d_array(X_val, self.input_shape), shuffle=False)
            if X_val is not None
            else None
        )
        self.training_history.clear()
        self.target_head = None
        self.target_shape = None
        started = time.perf_counter()
        for epoch in range(1, self.n_epochs + 1):
            epoch_started = time.perf_counter()
            self.model.train()
            total_loss = 0.0
            count = 0
            for (batch,) in train_loader:
                batch = batch.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                loss = torch.mean((self.model(batch) - batch) ** 2)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * batch.size(0)
                count += batch.size(0)
            row = {
                "epoch": float(epoch),
                "train_loss": total_loss / max(count, 1),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if val_loader is not None:
                self.model.eval()
                val_total = 0.0
                val_count = 0
                with torch.no_grad():
                    for (batch,) in val_loader:
                        batch = batch.to(self.device, non_blocking=True)
                        loss = torch.mean((self.model(batch) - batch) ** 2)
                        val_total += float(loss.detach().cpu()) * batch.size(0)
                        val_count += batch.size(0)
                row["val_loss"] = val_total / max(val_count, 1)
            self.training_history.append(row)
            if self.verbose and (epoch == 1 or epoch % 10 == 0 or epoch == self.n_epochs):
                print(f"[ConvAutoencoder] epoch={epoch:03d} train_loss={row['train_loss']:.6f}")

        if y is not None:
            y_array = np.asarray(y)
            if y_array.shape != np.asarray(X).shape:
                targets = _as_2d_targets(y)
                if len(targets) != len(X_train):
                    raise ValueError("X and y must contain the same number of samples")
                self.target_shape = tuple(y_array.shape[1:]) or (1,)
                self.target_head = Ridge(alpha=1.0, random_state=self.random_state)
                self.target_head.fit(self.encode(X_train), targets)
        self.is_fitted = True
        return self

    def _require_model(self) -> "_TorchConvAE":
        if self.model is None:
            raise ValueError("Model must be fitted before this operation")
        return self.model

    def encode(self, X: Any) -> np.ndarray:
        model = self._require_model()
        model.eval()
        outputs = []
        with torch.no_grad():
            for (batch,) in self._make_loader(_as_4d_array(X, self.input_shape), shuffle=False):
                outputs.append(model.encode(batch.to(self.device)).cpu().numpy())
        return np.vstack(outputs).astype(np.float32)

    def decode(self, z: Any) -> np.ndarray:
        model = self._require_model()
        latents = _as_2d_array(z)
        if latents.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent vectors of width {self.latent_dim}, "
                f"got {latents.shape[1]}"
            )
        model.eval()
        outputs = []
        loader = DataLoader(TensorDataset(torch.from_numpy(latents)), batch_size=self.batch_size)
        with torch.no_grad():
            for (batch,) in loader:
                outputs.append(model.decode(batch.to(self.device)).cpu().numpy())
        return np.vstack(outputs).astype(np.float32)

    def predict(self, X: Any) -> np.ndarray:
        if self.target_head is not None:
            prediction = np.asarray(self.target_head.predict(self.encode(X)), dtype=np.float32)
            if prediction.ndim == 1:
                prediction = prediction.reshape(-1, 1)
            if self.target_shape and self.target_shape != (1,):
                prediction = prediction.reshape(-1, *self.target_shape)
            return prediction
        return self.decode(self.encode(X))

    def reconstruct(self, X: Any) -> np.ndarray:
        return self.decode(self.encode(X))

    def reconstruction_metrics(
        self,
        X: Any,
        *,
        include_ssim: bool = False,
        image_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Optional[float]]:
        images = _as_4d_array(X, self.input_shape)
        reconstructed = self.reconstruct(images)
        metrics = reconstruction_metrics(
            images.reshape(len(images), -1),
            reconstructed.reshape(len(reconstructed), -1),
            include_ssim=include_ssim,
            image_shape=image_shape or self.input_shape,
        )
        metrics["latent_var_mean"] = float(np.var(self.encode(images), axis=0).mean())
        return metrics

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "config": {
                    "input_shape": self.input_shape,
                    "latent_dim": self.latent_dim,
                    "channels": self.channels,
                    "kernel_size": self.kernel_size,
                    "stride": self.stride,
                    "padding": self.padding,
                    "learning_rate": self.learning_rate,
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "random_state": self.random_state,
                },
                "model_state": self.model.state_dict() if self.model is not None else None,
                "target_head": self.target_head,
                "target_shape": self.target_shape,
                "training_history": self.training_history,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        config = payload["config"]
        integer_fields = (
            "latent_dim",
            "kernel_size",
            "stride",
            "padding",
            "n_epochs",
            "batch_size",
            "random_state",
        )
        for name in integer_fields:
            setattr(self, name, int(config[name]))
        self.input_shape = tuple(config["input_shape"])
        self.channels = tuple(config["channels"])
        self.learning_rate = float(config["learning_rate"])
        self._build()
        if payload.get("model_state") is not None:
            self._require_model().load_state_dict(payload["model_state"])
            self._require_model().eval()
        self.target_head = payload.get("target_head")
        self.target_shape = payload.get("target_shape")
        self.training_history = list(payload.get("training_history", []))
        self.is_fitted = bool(payload.get("is_fitted", False))


__all__ = ["ConvAutoencoderModel", "TORCH_AVAILABLE", "_as_4d_array"]
