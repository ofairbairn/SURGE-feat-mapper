"""All model adapter tests.

Covers: sklearn baselines, BoTorch GP, FT-Transformer, KAN, Vision,
PDE/operator models, Generative models (VAE/DDPM/CGAN), MLP Ensemble.

Each section:
  1. Registration check
  2. fit/predict shape test (tiny data, ≤3 epochs)
  3. predict_with_uncertainty where available
  4. edge-case / error tests where relevant

Per-test `pytest.importorskip` keeps sklearn tests runnable when torch
is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from surge.model.registry import MODEL_REGISTRY


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def reg_data():
    """60-train / 20-test 1-D regression."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6)).astype("float32")
    y = X[:, 0] * 2 - X[:, 1] + 0.1 * rng.standard_normal(80)
    return X[:60], y[:60].astype("float32"), X[60:], y[60:].astype("float32")


@pytest.fixture()
def clf_data():
    """75-train / 25-test binary classification."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 6)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:75], y[:75], X[75:], y[75:]


@pytest.fixture()
def multiout_data():
    """60-train / 20-test 2-column regression."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 6)).astype("float32")
    A = np.array([[1, -1], [0.5, 0.5], [-1, 0], [0, 1], [0.3, -0.3], [0.2, 0.8]])
    Y = (X @ A + 0.05 * rng.standard_normal((80, 2))).astype("float32")
    return X[:60], Y[:60], X[60:], Y[60:]


@pytest.fixture()
def unsup_data():
    """60-train / 20-test unsupervised data."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 6)).astype("float32")
    return X[:60], X[60:]


@pytest.fixture()
def owen_target_head_data():
    """Data where y shape differs from X so the latent Ridge head is used."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((80, 6)).astype("float32")
    A = np.array([[1, -1], [0.5, 0.5], [-1, 0], [0, 1], [0.3, -0.3], [0.2, 0.8]])
    Y = (X @ A + 0.05 * rng.standard_normal((80, 2))).astype("float32")
    return X[:60], Y[:60], X[60:], Y[60:]


# ── Sklearn baselines (Group C) ───────────────────────────────────────────────


def test_ridge_registered():
    assert "sklearn.ridge" in MODEL_REGISTRY


def test_ridge_fit_predict(reg_data):
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create("sklearn.ridge")
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_lgbm_regressor_registered():
    pytest.importorskip("lightgbm")
    assert "lgbm.regressor" in MODEL_REGISTRY


def test_lgbm_regressor_fit_predict(reg_data):
    pytest.importorskip("lightgbm")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create("lgbm.regressor", n_estimators=20)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)


def test_lgbm_classifier_registered():
    pytest.importorskip("lightgbm")
    assert "lgbm.classifier" in MODEL_REGISTRY


def test_lgbm_classifier_fit_predict_proba(clf_data):
    pytest.importorskip("lightgbm")
    X_tr, y_tr, X_te, _ = clf_data
    adapter = MODEL_REGISTRY.create("lgbm.classifier", n_estimators=20)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape == (len(X_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_catboost_regressor_registered():
    pytest.importorskip("catboost")
    assert "catboost.regressor" in MODEL_REGISTRY


def test_catboost_regressor_fit_predict(reg_data):
    pytest.importorskip("catboost")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create("catboost.regressor", iterations=20)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)


def test_catboost_classifier_registered():
    pytest.importorskip("catboost")
    assert "catboost.classifier" in MODEL_REGISTRY


def test_catboost_classifier_fit_predict_proba(clf_data):
    pytest.importorskip("catboost")
    X_tr, y_tr, X_te, _ = clf_data
    adapter = MODEL_REGISTRY.create("catboost.classifier", iterations=20)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape == (len(X_te), 2)


# ── BoTorch GP (Group D) ──────────────────────────────────────────────────────


def test_botorch_gp_registered():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    assert "botorch.gp" in MODEL_REGISTRY


def test_botorch_gp_fit_predict():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype("float32")
    y = (X[:, 0] * 2).astype("float32")
    X_tr, y_tr, X_te = X[:24], y[:24], X[24:]

    adapter = MODEL_REGISTRY.create("botorch.gp", n_train_iter=5)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (6,)
    assert np.isfinite(preds).all()


def test_botorch_gp_predict_with_uncertainty():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype("float32")
    y = (X[:, 0] * 2).astype("float32")
    X_tr, y_tr, X_te = X[:24], y[:24], X[24:]

    adapter = MODEL_REGISTRY.create("botorch.gp", n_train_iter=5)
    adapter.fit(X_tr, y_tr)
    mean, std = adapter.predict_with_uncertainty(X_te)
    assert mean.shape == (6,)
    assert std.shape == (6,)
    assert (std >= 0).all()


def test_botorch_gp_size_guard():
    """ExactGP raises ValueError for n > 5000."""
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6001, 4)).astype("float32")
    y = rng.standard_normal(6001).astype("float32")
    adapter = MODEL_REGISTRY.create("botorch.gp")
    with pytest.raises(ValueError):
        adapter.fit(X, y)


def test_botorch_sparse_gp_registered():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    assert "botorch.sparse_gp" in MODEL_REGISTRY


def test_botorch_sparse_gp_fit_predict():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4)).astype("float32")
    y = (X[:, 0]).astype("float32")
    X_tr, y_tr, X_te = X[:60], y[:60], X[60:]
    adapter = MODEL_REGISTRY.create(
        "botorch.sparse_gp", n_train_iter=5, n_inducing=10
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (20,)


# ── FT-Transformer (Group E) ──────────────────────────────────────────────────


def test_ft_transformer_registered():
    pytest.importorskip("torch")
    assert "pytorch.ft_transformer" in MODEL_REGISTRY


def test_ft_transformer_classifier_registered():
    pytest.importorskip("torch")
    assert "pytorch.ft_transformer_classifier" in MODEL_REGISTRY


def test_ft_transformer_fit_predict(reg_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.ft_transformer",
        n_epochs=2, d_model=16, n_heads=2, n_layers=1, batch_size=16,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_ft_transformer_classifier_fit_predict_proba(clf_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = clf_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.ft_transformer_classifier",
        n_epochs=2, d_model=16, n_heads=2, n_layers=1, batch_size=16,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape == (len(X_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ── KAN (Group E) ─────────────────────────────────────────────────────────────


def test_kan_registered():
    pytest.importorskip("torch")
    pytest.importorskip("efficient_kan")
    assert "pytorch.kan" in MODEL_REGISTRY


def test_kan_classifier_registered():
    pytest.importorskip("torch")
    pytest.importorskip("efficient_kan")
    assert "pytorch.kan_classifier" in MODEL_REGISTRY


def test_kan_fit_predict(reg_data):
    pytest.importorskip("torch")
    pytest.importorskip("efficient_kan")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.kan",
        n_epochs=2, hidden_dims=[16, 16], grid_size=3, batch_size=16,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_kan_classifier_fit_predict(clf_data):
    pytest.importorskip("torch")
    pytest.importorskip("efficient_kan")
    X_tr, y_tr, X_te, _ = clf_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.kan_classifier",
        n_epochs=2, hidden_dims=[16, 16], grid_size=3, batch_size=16,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert set(np.unique(preds)).issubset({0, 1})


# ── Vision models (Group F) ───────────────────────────────────────────────────


@pytest.fixture()
def img_data():
    """30 samples of 32×32×3 flat images, 10 classes."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3072)).astype("float32")
    y = rng.integers(0, 10, 30)
    return X[:24], y[:24], X[24:], y[24:]


def test_alexnet_registered():
    pytest.importorskip("torch")
    assert "pytorch.alexnet" in MODEL_REGISTRY


def test_alexnet_fit_predict(img_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = img_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.alexnet", n_epochs=1, n_classes=10, batch_size=8,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert set(np.unique(preds)).issubset(set(range(10)))
    assert proba.shape == (len(X_te), 10)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_vit_registered():
    pytest.importorskip("torch")
    assert "pytorch.vit" in MODEL_REGISTRY


def test_vit_fit_predict(img_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = img_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.vit",
        n_epochs=1, n_classes=10, d_model=16, n_heads=2, n_layers=1,
        patch_size=4, batch_size=8,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape == (len(X_te), 10)


def test_resnet20_registered():
    pytest.importorskip("torch")
    assert "pytorch.resnet20" in MODEL_REGISTRY


def test_resnet20_fit_predict(img_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = img_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.resnet20", n_epochs=1, n_classes=10, batch_size=8,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    proba = adapter.predict_proba(X_te)
    assert preds.shape == (len(X_te),)
    assert proba.shape == (len(X_te), 10)


def test_resnet56_registered():
    pytest.importorskip("torch")
    assert "pytorch.resnet56" in MODEL_REGISTRY


def test_resnet56_fit_predict(img_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = img_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.resnet56", n_epochs=1, n_classes=10, batch_size=8,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)


def test_mnist_cnn_registered():
    pytest.importorskip("torch")
    assert "pytorch.mnist_cnn" in MODEL_REGISTRY


def test_mnist_cnn_analyzes_precomputed_predictions(capsys):
    pytest.importorskip("torch")
    from surge.model.backends.mnist_cnn import MNISTCNNModel
    from surge.metrics import expected_calibration_error

    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2, 1, 0])
    y_prob = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.30, 0.60, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.80, 0.10],
            [0.60, 0.30, 0.10],
            [0.10, 0.10, 0.80],
            [0.10, 0.70, 0.20],
            [0.60, 0.10, 0.30],
        ]
    )

    analysis = MNISTCNNModel.analyze_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        labels=[0, 1, 2],
        class_names=["Alpha", "Beta", "Gamma"],
        top_k=3,
    )

    assert analysis["per_class"][0]["accuracy"] == pytest.approx(100 / 3)
    assert analysis["per_class"][0]["ece"] == pytest.approx(
        expected_calibration_error((y_true == 0).astype(int), y_prob[:, 0])
    )
    assert analysis["hardest_class"]["name"] == "Alpha"
    assert analysis["top_confusions"][0] == {
        "true_label": 0,
        "true_name": "Alpha",
        "predicted_label": 1,
        "predicted_name": "Beta",
        "count": 2,
        "sample_indices": [1, 2],
    }
    assert analysis["misclassified_indices"] == [1, 2, 4, 6, 7]
    output = capsys.readouterr().out
    assert "Accuracy of Alpha: 33.33 %" in output
    assert "ECE of Alpha (one-vs-rest):" in output
    assert "Most confused: Alpha -> Beta: 2 samples" in output


def test_mnist_cnn_fit_predict_and_roundtrip(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(42)

    n_samples = 100
    X = rng.integers(0, 256, size=(n_samples, 28 * 28), dtype=np.uint8)
    y = np.array([10 + (i % 10) for i in range(n_samples)], dtype=np.int64)

    n_train = 80
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    adapter = MODEL_REGISTRY.create(
        "pytorch.mnist_cnn",
        epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        device="cpu",
    )
    adapter.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    assert hasattr(adapter, "training_history")
    assert isinstance(adapter.training_history, list)
    assert len(adapter.training_history) > 0

    for entry in adapter.training_history:
        assert "epoch" in entry
        assert "train_loss" in entry
        assert "val_accuracy" in entry
        assert "val_loss" in entry

    preds = adapter.predict(X[:5])
    assert preds.shape[0] == 5
    assert np.issubdtype(preds.dtype, np.integer)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))

    probs = adapter.predict_proba(X[:5])
    assert probs.shape == (5, 10)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    class_names = [f"class-{label}" for label in range(10, 20)]
    analysis = adapter.analyze_classification(
        X_val,
        y_val,
        class_names=class_names,
        top_k=3,
        print_report=False,
    )
    assert [result["label"] for result in analysis["per_class"]] == list(range(10, 20))
    assert [result["name"] for result in analysis["per_class"]] == class_names
    assert sum(result["total"] for result in analysis["per_class"]) == len(y_val)
    assert analysis["hardest_class"] in analysis["per_class"]
    assert len(analysis["top_confusions"]) <= 3
    assert analysis["misclassified_indices"] == np.where(adapter.predict(X_val) != y_val)[0].tolist()

    path = tmp_path / "mnist_cnn_model.pt"
    adapter.save(path)

    restored = MODEL_REGISTRY.create(
        "pytorch.mnist_cnn",
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        device="cpu",
    )
    restored.load(path)
    restored_preds = restored.predict(X[:3])
    assert restored_preds.shape[0] == 3

    adapter.fit(X_train[:20], y_train[:20], X_val=X_val[:5], y_val=y_val[:5])
    assert len(adapter.training_history) > 0


def test_mnist_cnn_supports_channel_last_and_custom_input_size():
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)

    X = rng.integers(0, 256, size=(24, 64, 64, 1), dtype=np.uint8)
    y = np.array([i % 6 for i in range(24)], dtype=np.int64)

    X_train, X_val = X[:18], X[18:]
    y_train, y_val = y[:18], y[18:]

    adapter = MODEL_REGISTRY.create(
        "pytorch.mnist_cnn",
        input_size=(64, 64),
        input_channels=1,
        hidden_channels=(16, 32),
        epochs=2,
        batch_size=6,
        learning_rate=1e-3,
        device="cpu",
    )
    adapter.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    preds = adapter.predict(X_val)
    probs = adapter.predict_proba(X_val)

    assert preds.shape[0] == X_val.shape[0]
    assert probs.shape == (X_val.shape[0], len(np.unique(y_train)))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# ── PDE / operator models (Group G) ──────────────────────────────────────────


def test_fno2d_registered():
    pytest.importorskip("torch")
    assert "pytorch.fno2d" in MODEL_REGISTRY


def test_fno2d_fit_predict():
    """FNO2D on 2D spatial fields (B, nx, ny)."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 8, 8)).astype("float32")
    y = X + 0.1 * rng.standard_normal((20, 8, 8)).astype("float32")
    X_tr, y_tr, X_te, y_te = X[:16], y[:16], X[16:], y[16:]

    adapter = MODEL_REGISTRY.create(
        "pytorch.fno2d",
        n_epochs=2, hidden_channels=8, n_modes=4, n_layers=1, batch_size=4,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape[0] == len(X_te)
    assert np.isfinite(preds).all()


def test_unet_registered():
    pytest.importorskip("torch")
    assert "pytorch.unet" in MODEL_REGISTRY


def test_unet_fit_predict():
    """U-Net on 2D fields (B, C, H, W)."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 1, 16, 16)).astype("float32")
    y = X + 0.1 * rng.standard_normal((20, 1, 16, 16)).astype("float32")
    X_tr, y_tr, X_te, y_te = X[:16], y[:16], X[16:], y[16:]

    adapter = MODEL_REGISTRY.create(
        "pytorch.unet",
        n_epochs=2, base_channels=8, depth=2, batch_size=4,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape[0] == len(X_te)
    assert np.isfinite(preds).all()


# ── Generative / conditional models (Group H) ─────────────────────────────────


def test_vae_registered():
    pytest.importorskip("torch")
    assert "pytorch.vae" in MODEL_REGISTRY


def test_vae_fit_predict(reg_data):
    """VAE: fit(X, y) → predict(X) returns ŷ (regression surrogate)."""
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.vae",
        latent_dim=4, hidden_dim=16, n_epochs=2, batch_size=8, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_vae_predict_with_uncertainty(reg_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.vae",
        latent_dim=4, hidden_dim=16, n_epochs=2, batch_size=8, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    mean, std = adapter.predict_with_uncertainty(X_te)
    assert mean.shape == (len(X_te),)
    assert std.shape == (len(X_te),)


def test_ddpm_registered():
    pytest.importorskip("torch")
    assert "pytorch.ddpm" in MODEL_REGISTRY


def test_ddpm_fit_predict():
    """DDPM: conditional field-to-field diffusion model."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 16)).astype("float32")
    y = X + 0.1 * rng.standard_normal((40, 16)).astype("float32")
    X_tr, y_tr, X_te, y_te = X[:30], y[:30], X[30:], y[30:]

    adapter = MODEL_REGISTRY.create(
        "pytorch.ddpm",
        n_timesteps=5, hidden_channels=8, n_epochs=2, batch_size=8, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == y_te.shape
    assert np.isfinite(preds).all()


def test_cgan_registered():
    pytest.importorskip("torch")
    assert "pytorch.cgan" in MODEL_REGISTRY


def test_cgan_fit_predict():
    """CGAN: conditional GAN field-to-field surrogate."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 16)).astype("float32")
    y = X + 0.1 * rng.standard_normal((40, 16)).astype("float32")
    X_tr, y_tr, X_te, y_te = X[:30], y[:30], X[30:], y[30:]

    adapter = MODEL_REGISTRY.create(
        "pytorch.cgan",
        latent_dim=4, hidden_channels=16, n_gen_layers=2, n_disc_layers=2,
        n_epochs=2, batch_size=8, n_predict_samples=2,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == y_te.shape
    assert np.isfinite(preds).all()


def test_autoencoder_registered():
    pytest.importorskip("torch")
    assert "pytorch.autoencoder" in MODEL_REGISTRY


def test_autoencoder_fit_predict_reconstruction_shape(unsup_data):
    pytest.importorskip("torch")
    X_tr, X_te = unsup_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.autoencoder",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr)
    recon = adapter.predict(X_te)
    assert recon.shape == X_te.shape
    assert np.isfinite(recon).all()


def test_autoencoder_encode_decode_roundtrip(unsup_data):
    pytest.importorskip("torch")
    X_tr, X_te = unsup_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.autoencoder",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr)
    z = adapter.encode(X_te)
    assert z.shape == (len(X_te), 4)

    recon = adapter.decode(z)
    assert recon.shape == X_te.shape


def test_autoencoder_target_head_when_y_shape_differs(owen_target_head_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, y_te = owen_target_head_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.autoencoder",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == y_te.shape


def test_autoencoder_reconstruction_metrics_basic(unsup_data):
    pytest.importorskip("torch")
    X_tr, X_te = unsup_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.autoencoder",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr)
    metrics = adapter.reconstruction_metrics(X_te)
    for key in ("mse", "mae", "rmse", "latent_var_mean"):
        assert key in metrics
        assert metrics[key] is not None
        assert np.isfinite(metrics[key])


def test_autoencoder_save_load_round_trip(unsup_data, tmp_path):
    pytest.importorskip("torch")
    X_tr, X_te = unsup_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.autoencoder",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr)
    preds_before = adapter.predict(X_te)

    save_path = tmp_path / "autoencoder.joblib"
    adapter.save(save_path)

    loaded = MODEL_REGISTRY.create("pytorch.autoencoder", latent_dim=4, hidden_dims=(16, 8))
    loaded.load(save_path)
    preds_after = loaded.predict(X_te)
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5, atol=1e-6)


def test_owen_vae_registered():
    pytest.importorskip("torch")
    assert "pytorch.owen_vae" in MODEL_REGISTRY


def test_owen_vae_fit_predict_reconstruction_shape(unsup_data):
    pytest.importorskip("torch")
    X_tr, X_te = unsup_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.owen_vae",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr)
    recon = adapter.predict(X_te)
    assert recon.shape == X_te.shape
    assert np.isfinite(recon).all()


def test_owen_vae_fit_with_validation_data(unsup_data):
    pytest.importorskip("torch")
    X_tr, X_val = unsup_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.owen_vae",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr, X_val=X_val)
    history = adapter.training_history
    assert history is not None
    assert len(history) == 2
    assert all("train_loss" in row for row in history)
    assert all("val_loss" in row for row in history)


def test_owen_vae_encode_decode_roundtrip(unsup_data):
    pytest.importorskip("torch")
    X_tr, X_te = unsup_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.owen_vae",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr)
    z = adapter.encode(X_te, sample=False)
    assert z.shape == (len(X_te), 4)
    assert np.isfinite(z).all()

    recon = adapter.decode(z)
    assert recon.shape == X_te.shape
    assert np.isfinite(recon).all()


def test_owen_vae_target_head_when_y_shape_differs(owen_target_head_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, y_te = owen_target_head_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.owen_vae",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == y_te.shape
    assert np.isfinite(preds).all()


def test_owen_vae_reconstruction_metrics_basic(unsup_data):
    pytest.importorskip("torch")
    X_tr, X_te = unsup_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.owen_vae",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr)
    metrics = adapter.reconstruction_metrics(X_te)
    for key in ("mse", "mae", "rmse", "latent_var_mean", "kl"):
        assert key in metrics
        assert metrics[key] is not None
        assert np.isfinite(metrics[key])
    assert metrics["ssim"] is None


def test_owen_vae_reconstruction_metrics_ssim_requires_shape(unsup_data):
    pytest.importorskip("torch")
    from surge.model.backends.owen_vae import reconstruction_metrics

    _, X_te = unsup_data
    with pytest.raises(ValueError):
        reconstruction_metrics(X_te, X_te, include_ssim=True)


def test_owen_vae_save_load_round_trip(unsup_data, tmp_path):
    pytest.importorskip("torch")
    X_tr, X_te = unsup_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.owen_vae",
        latent_dim=4,
        hidden_dims=(16, 8),
        n_epochs=2,
        batch_size=16,
        random_state=0,
    )
    adapter.fit(X_tr)
    preds_before = adapter.predict(X_te)

    save_path = tmp_path / "owen_vae.joblib"
    adapter.save(save_path)

    loaded_adapter = MODEL_REGISTRY.create(
        "pytorch.owen_vae",
        latent_dim=4,
        hidden_dims=(16, 8),
    )
    loaded_adapter.load(save_path)
    preds_after = loaded_adapter.predict(X_te)
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5, atol=1e-6)


def test_owen_vae_predict_before_fit_raises():
    pytest.importorskip("torch")
    adapter = MODEL_REGISTRY.create("pytorch.owen_vae", latent_dim=4, hidden_dims=(16, 8))
    with pytest.raises(ValueError):
        adapter.predict(np.zeros((5, 6), dtype="float32"))


# ── MLP Ensemble (Group I) ────────────────────────────────────────────────────


def test_mlp_ensemble_registered():
    pytest.importorskip("torch")
    assert "pytorch.mlp_ensemble" in MODEL_REGISTRY


def test_mlp_ensemble_fit_predict(reg_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.mlp_ensemble",
        n_ensembles=2, n_epochs=3, hidden_dim=16, n_layers=2,
        batch_size=16, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert np.isfinite(preds).all()


def test_mlp_ensemble_predict_with_uncertainty(reg_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = reg_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.mlp_ensemble",
        n_ensembles=2, n_epochs=3, hidden_dim=16, n_layers=2,
        batch_size=16, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    mean, std = adapter.predict_with_uncertainty(X_te)
    assert mean.shape == (len(X_te),)
    assert std.shape == (len(X_te),)
    assert (std >= 0).all()


def test_mlp_ensemble_multioutput(multiout_data):
    pytest.importorskip("torch")
    X_tr, y_tr, X_te, _ = multiout_data
    adapter = MODEL_REGISTRY.create(
        "pytorch.mlp_ensemble",
        n_ensembles=2, n_epochs=3, hidden_dim=16, n_layers=2,
        batch_size=16, patience=0,
    )
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict(X_te)
    mean, std = adapter.predict_with_uncertainty(X_te)
    assert preds.shape == (len(X_te), 2)
    assert mean.shape == (len(X_te), 2)
    assert std.shape == (len(X_te), 2)


# ── Benchmark smoke tests ─────────────────────────────────────────────────────


def test_ridge_benchmark_smoke():
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark(
        "tabular.california_housing", model_key="sklearn.ridge", seed=0
    )
    assert result is not None
    assert "test_r2" in result.metrics


def test_lgbm_classifier_benchmark_smoke():
    pytest.importorskip("lightgbm")
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark("tabular.iris", model_key="lgbm.classifier", seed=0)
    assert result is not None
    assert "test_accuracy" in result.metrics


@pytest.mark.slow
def test_ft_transformer_benchmark_smoke():
    pytest.importorskip("torch")
    from surge.benchmarks.registry import run_benchmark

    result = run_benchmark(
        "tabular.diabetes", model_key="pytorch.ft_transformer", seed=0
    )
    assert result is not None
    assert "test_r2" in result.metrics
