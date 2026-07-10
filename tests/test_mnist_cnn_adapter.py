import numpy as np

from surge.model.adapters.mnist_cnn import MNISTCNNAdapter


def test_mnist_cnn_adapter_fit_predict_and_roundtrip(tmp_path):
    """Test that MNISTCNNAdapter can fit, predict, save, and load with validation data."""
    rng = np.random.default_rng(42)

    n_samples = 100
    X = rng.integers(0, 256, size=(n_samples, 28 * 28), dtype=np.uint8)
    y = np.array([10 + (i % 10) for i in range(n_samples)], dtype=np.int64)

    n_train = 80
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    adapter = MNISTCNNAdapter(
        epochs=10,
        batch_size=8,
        learning_rate=1e-2,
        device="cpu",
    )
    
    adapter.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    assert hasattr(adapter, "training_history")
    assert isinstance(adapter.training_history, list)
    assert len(adapter.training_history) > 0, "training_history should have entries after fit"

    for entry in adapter.training_history:
        assert "epoch" in entry, f"Entry {entry} missing 'epoch' key"
        assert "train_loss" in entry, f"Entry {entry} missing 'train_loss' key"
        assert "val_accuracy" in entry, f"Entry {entry} missing 'val_accuracy' key"
        assert "val_loss" in entry, f"Entry {entry} missing 'val_loss' key"

    preds = adapter.predict(X[:5])
    assert preds.shape[0] == 5
    assert np.issubdtype(preds.dtype, np.integer)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))

    probs = adapter.predict_proba(X[:5])
    assert probs.shape == (5, 10)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)

    path = tmp_path / "mnist_cnn_model.pt"
    adapter.save(path)

    restored = MNISTCNNAdapter(epochs=1, batch_size=2, learning_rate=1e-3, device="cpu")
    restored.load(path)
    restored_preds = restored.predict(X[:3])
    assert restored_preds.shape[0] == 3

    adapter.fit(X_train[:20], y_train[:20], X_val=X_val[:5], y_val=y_val[:5])
    assert len(adapter.training_history) > 0, "training_history should be repopulated after second fit"


def test_mnist_cnn_adapter_supports_channel_last_and_custom_input_size():
    """Validate NHWC support and non-28x28 input sizes."""
    rng = np.random.default_rng(7)

    # Synthetic 64x64 grayscale images in channel-last format.
    X = rng.integers(0, 256, size=(24, 64, 64, 1), dtype=np.uint8)
    y = np.array([i % 6 for i in range(24)], dtype=np.int64)

    X_train, X_val = X[:18], X[18:]
    y_train, y_val = y[:18], y[18:]

    adapter = MNISTCNNAdapter(
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
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)

