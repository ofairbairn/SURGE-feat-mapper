"""Progress and validation-history tests for unsupervised backends."""

from typing import Any

import numpy as np
import pytest


class _FakeEpochProgress:
    def __init__(self, start: int, stop: int) -> None:
        self.values = range(start, stop)
        self.postfixes: list[dict[str, Any]] = []

    def __iter__(self):
        return iter(self.values)

    def set_postfix(self, values: dict[str, Any]) -> None:
        self.postfixes.append(values)


class _FakeFitProgress:
    def __init__(self) -> None:
        self.updates = 0
        self.postfix: dict[str, Any] = {}

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, amount: int) -> None:
        self.updates += amount

    def set_postfix(self, **values: Any) -> None:
        self.postfix = values


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    return (
        rng.normal(size=(24, 6)).astype(np.float32),
        rng.normal(size=(8, 6)).astype(np.float32),
    )


def test_autoencoder_records_validation_loss_and_timed_tqdm_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from surge.model.backends import autoencoder as backend

    progress = _FakeEpochProgress(1, 3)
    monkeypatch.setattr(backend, "TQDM_AVAILABLE", True)
    monkeypatch.setattr(backend, "trange", lambda *args, **kwargs: progress)
    X_train, X_val = _training_data()
    model = backend.AutoencoderModel(
        latent_dim=2,
        hidden_dims=(8,),
        n_epochs=2,
        batch_size=8,
        device="cpu",
        verbose=True,
    )

    model.fit(X_train, X_val=X_val)

    assert len(model.training_history) == 2
    assert all("val_loss" in row for row in model.training_history)
    assert all(row["epoch_seconds"] >= 0 for row in model.training_history)
    assert all(row["elapsed_seconds"] >= 0 for row in model.training_history)
    assert len(progress.postfixes) == 2
    assert all("val_loss" in postfix for postfix in progress.postfixes)


def test_owen_vae_reports_timed_tqdm_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from surge.model.backends import owen_vae as backend

    progress = _FakeEpochProgress(1, 3)
    monkeypatch.setattr(backend, "TQDM_AVAILABLE", True)
    monkeypatch.setattr(backend, "trange", lambda *args, **kwargs: progress)
    X_train, X_val = _training_data()
    model = backend.OwenVAEModel(
        latent_dim=2,
        hidden_dims=(8,),
        n_epochs=2,
        batch_size=8,
        device="cpu",
        verbose=True,
    )

    model.fit(X_train, X_val=X_val)

    assert all("val_loss" in row for row in model.training_history)
    assert all("epoch_seconds" in row for row in model.training_history)
    assert all("elapsed_seconds" in row for row in model.training_history)
    assert len(progress.postfixes) == 2
    assert all({"recon", "kl", "val_loss"} <= postfix.keys() for postfix in progress.postfixes)


def test_pca_uses_single_fit_progress_indicator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surge.model.backends import pca as backend

    progress = _FakeFitProgress()
    monkeypatch.setattr(backend, "TQDM_AVAILABLE", True)
    monkeypatch.setattr(backend, "tqdm", lambda **kwargs: progress)
    X_train, _ = _training_data()

    backend.PCAModel(n_components=2, verbose=True).fit(X_train)

    assert progress.updates == 1
    assert progress.postfix["components"] == 2
    assert "elapsed" in progress.postfix
