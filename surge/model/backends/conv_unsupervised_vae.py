"""Convolutional unsupervised variational autoencoder backend for SURGE."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from .conv_autoencoder import (
    TORCH_AVAILABLE,
    ConvAutoencoderModel,
    ImageShape,
    _as_2d_targets,
    _as_4d_array,
)
from .unsupervised_vae import (
    _LOGVAR_MAX,
    _LOGVAR_MIN,
    _kl_divergence_from_gaussian_params,
    reconstruction_metrics,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from .conv_autoencoder import _TorchConvAE

    class _TorchConvVAE(_TorchConvAE):
        """Convolutional VAE with Gaussian parameters at the vector bottleneck."""

        def __init__(
            self,
            input_shape: ImageShape,
            latent_dim: int,
            channels: Sequence[int],
            kernel_size: int,
            stride: int,
            padding: int,
        ) -> None:
            super().__init__(
                input_shape,
                latent_dim,
                channels,
                kernel_size,
                stride,
                padding,
            )
            flat_dim = self.encoder_fc.in_features
            del self.encoder_fc
            self.fc_mu = nn.Linear(flat_dim, latent_dim)
            self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        def encode(self, x):
            encoded = self.encoder_conv(x).flatten(1)
            return self.fc_mu(encoded), self.fc_logvar(encoded)

        def reparameterize(self, mu, logvar):
            bounded_logvar = torch.clamp(logvar, min=_LOGVAR_MIN, max=_LOGVAR_MAX)
            std = torch.exp(0.5 * bounded_logvar)
            return mu + torch.randn_like(std) * std

        def forward(self, x):
            mu, logvar = self.encode(x)
            reconstruction = self.decode(self.reparameterize(mu, logvar))
            return reconstruction, mu, logvar


class ConvUnsupervisedVAEModel(ConvAutoencoderModel):
    """Convolutional beta-VAE for unsupervised image representation learning."""

    def __init__(
        self,
        *,
        input_shape: Tuple[int, int, int],
        latent_dim: int = 8,
        channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        learning_rate: float = 1e-3,
        n_epochs: int = 150,
        batch_size: int = 128,
        beta: float = 1.0,
        max_grad_norm: Optional[float] = 1.0,
        random_state: int = 42,
        device: Optional[str] = None,
        dataloader_num_workers: int = 0,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            latent_dim=latent_dim,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
            dataloader_num_workers=dataloader_num_workers,
            verbose=verbose,
            **kwargs,
        )
        self.beta = float(beta)
        if self.beta < 0:
            raise ValueError("beta cannot be negative")
        if max_grad_norm is not None and float(max_grad_norm) <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        self.max_grad_norm = (
            None if max_grad_norm is None else float(max_grad_norm)
        )

    def _build(self) -> None:
        torch.manual_seed(self.random_state)
        self.model = _TorchConvVAE(
            self.input_shape,
            self.latent_dim,
            self.channels,
            self.kernel_size,
            self.stride,
            self.padding,
        ).to(self.device)

    def _loss_components(self, reconstruction, target, mu, logvar):
        bounded_logvar = torch.clamp(logvar, min=_LOGVAR_MIN, max=_LOGVAR_MAX)
        reconstruction_loss = torch.mean((reconstruction - target) ** 2)
        kl = -0.5 * torch.mean(
            1.0 + bounded_logvar - mu.pow(2) - bounded_logvar.exp()
        )
        return reconstruction_loss + self.beta * kl, reconstruction_loss, kl

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        X_val: Any = None,
        y_val: Any = None,
        finetune: bool = False,
    ) -> "ConvUnsupervisedVAEModel":
        del y_val
        X_train = _as_4d_array(X, self.input_shape)
        if self.model is None or not finetune:
            self._build()
        model = self._require_model()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        train_loader = self._make_loader(X_train, shuffle=True)
        val_loader = (
            self._make_loader(_as_4d_array(X_val, self.input_shape), shuffle=False)
            if X_val is not None
            else None
        )

        self.training_history.clear()
        self.target_head = None
        self.target_shape = None
        training_started = time.perf_counter()
        for epoch in range(1, self.n_epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
            sample_count = 0
            for (batch,) in train_loader:
                batch = batch.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                reconstruction, mu, logvar = model(batch)
                loss, recon_loss, kl = self._loss_components(
                    reconstruction, batch, mu, logvar
                )
                values = (reconstruction, mu, logvar, loss, recon_loss, kl)
                if not all(torch.isfinite(value).all() for value in values):
                    raise FloatingPointError(
                        "Convolutional VAE produced a non-finite training value "
                        f"at epoch={epoch}; check input scale, beta, and learning_rate"
                    )
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError(
                        f"Convolutional VAE produced a non-finite gradient at epoch={epoch}"
                    )
                optimizer.step()
                batch_size = batch.size(0)
                sample_count += batch_size
                totals["loss"] += float(loss.detach().cpu().item()) * batch_size
                totals["recon"] += float(recon_loss.detach().cpu().item()) * batch_size
                totals["kl"] += float(kl.detach().cpu().item()) * batch_size

            row: Dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": totals["loss"] / max(sample_count, 1),
                "train_recon": totals["recon"] / max(sample_count, 1),
                "train_kl": totals["kl"] / max(sample_count, 1),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "elapsed_seconds": time.perf_counter() - training_started,
            }
            if val_loader is not None:
                model.eval()
                val_totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
                val_count = 0
                with torch.no_grad():
                    for (batch,) in val_loader:
                        batch = batch.to(self.device, non_blocking=True)
                        reconstruction, mu, logvar = model(batch)
                        loss, recon_loss, kl = self._loss_components(
                            reconstruction, batch, mu, logvar
                        )
                        values = (reconstruction, mu, logvar, loss, recon_loss, kl)
                        if not all(torch.isfinite(value).all() for value in values):
                            raise FloatingPointError(
                                "Convolutional VAE produced a non-finite validation value "
                                f"at epoch={epoch}"
                            )
                        batch_size = batch.size(0)
                        val_count += batch_size
                        val_totals["loss"] += float(loss.cpu().item()) * batch_size
                        val_totals["recon"] += float(recon_loss.cpu().item()) * batch_size
                        val_totals["kl"] += float(kl.cpu().item()) * batch_size
                row.update(
                    {
                        "val_loss": val_totals["loss"] / max(val_count, 1),
                        "val_recon": val_totals["recon"] / max(val_count, 1),
                        "val_kl": val_totals["kl"] / max(val_count, 1),
                    }
                )
            self.training_history.append(row)
            if self.verbose and (
                epoch == 1 or epoch % 10 == 0 or epoch == self.n_epochs
            ):
                print(
                    f"[ConvUnsupervisedVAE] epoch={epoch:03d} "
                    f"train_loss={row['train_loss']:.6f} "
                    f"recon={row['train_recon']:.6f} kl={row['train_kl']:.6f}"
                )

        if y is not None:
            y_array = np.asarray(y)
            if y_array.shape != np.asarray(X).shape:
                targets = _as_2d_targets(y)
                if len(targets) != len(X_train):
                    raise ValueError("X and y must contain the same number of samples")
                self.target_shape = tuple(y_array.shape[1:]) or (1,)
                self.target_head = Ridge(alpha=1.0, random_state=self.random_state)
                self.target_head.fit(self.encode(X_train, sample=False), targets)
        self.is_fitted = True
        return self

    def encode(self, X: Any, *, sample: bool = False) -> np.ndarray:
        model = self._require_model()
        model.eval()
        latents = []
        images = _as_4d_array(X, self.input_shape)
        with torch.no_grad():
            for (batch,) in self._make_loader(images, shuffle=False):
                mu, logvar = model.encode(batch.to(self.device, non_blocking=True))
                latent = model.reparameterize(mu, logvar) if sample else mu
                latents.append(latent.cpu().numpy())
        return np.vstack(latents).astype(np.float32)

    def predict(self, X: Any) -> np.ndarray:
        if self.target_head is not None:
            prediction = np.asarray(
                self.target_head.predict(self.encode(X, sample=False)),
                dtype=np.float32,
            )
            if prediction.ndim == 1:
                prediction = prediction.reshape(-1, 1)
            if self.target_shape and self.target_shape != (1,):
                prediction = prediction.reshape(-1, *self.target_shape)
            return prediction
        return self.reconstruct(X, sample=False)

    def reconstruct(self, X: Any, *, sample: bool = False) -> np.ndarray:
        return self.decode(self.encode(X, sample=sample))

    def reconstruction_metrics(
        self,
        X: Any,
        *,
        include_ssim: bool = False,
        image_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Optional[float]]:
        images = _as_4d_array(X, self.input_shape)
        reconstructed = self.reconstruct(images, sample=False)
        metrics = reconstruction_metrics(
            images.reshape(len(images), -1),
            reconstructed.reshape(len(reconstructed), -1),
            include_ssim=include_ssim,
            image_shape=image_shape or self.input_shape,
        )
        model = self._require_model()
        model.eval()
        mus = []
        logvars = []
        with torch.no_grad():
            for (batch,) in self._make_loader(images, shuffle=False):
                mu, logvar = model.encode(batch.to(self.device, non_blocking=True))
                mus.append(mu.cpu().numpy())
                logvars.append(logvar.cpu().numpy())
        mu_array = np.vstack(mus)
        logvar_array = np.vstack(logvars)
        metrics["latent_var_mean"] = float(np.var(mu_array, axis=0).mean())
        metrics["kl"] = _kl_divergence_from_gaussian_params(
            mu_array, logvar_array
        )
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
                    "beta": self.beta,
                    "max_grad_norm": self.max_grad_norm,
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
        self.beta = float(config["beta"])
        saved_max_grad_norm = config.get("max_grad_norm")
        self.max_grad_norm = (
            None
            if saved_max_grad_norm is None
            else float(saved_max_grad_norm)
        )
        super().load(path)


__all__ = [
    "ConvUnsupervisedVAEModel",
    "TORCH_AVAILABLE",
    "_as_4d_array",
]
