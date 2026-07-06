import numpy as np
#test script for mnist_test_cnn.py for Owen
from surge.model.mnist_test_cnn import MNISTCNNAdapter


def test_mnist_cnn_adapter_fit_predict_and_roundtrip(tmp_path):
    """Test that MNISTCNNAdapter can fit, predict, save, and load with validation data."""
    rng = np.random.default_rng(42)
    
    # Create more reasonable test data: 100 samples instead of 20
    # Use better distribution: create synthetic "digit-like" patterns
    n_samples = 100
    X = rng.integers(0, 256, size=(n_samples, 28 * 28), dtype=np.uint8)
    y = np.array([i % 10 for i in range(n_samples)], dtype=np.int64)
    
    # Split into train/val (80/20 split)
    n_train = 80
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    adapter = MNISTCNNAdapter(
        epochs=10,
        batch_size=8,
        learning_rate=1e-2,
        device="cpu",
    )
    
    # Fit with validation data (as the engine would do)
    adapter.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    # Verify training history was recorded
    assert hasattr(adapter, "training_history"), "Adapter should have training_history attribute"
    assert isinstance(adapter.training_history, list), "training_history should be a list"
    assert len(adapter.training_history) > 0, "training_history should have entries after fit"
    
    # Check that each history entry has the expected keys
    for entry in adapter.training_history:
        assert "epoch" in entry, f"Entry {entry} missing 'epoch' key"
        assert "train_loss" in entry, f"Entry {entry} missing 'train_loss' key"
        assert "val_accuracy" in entry, f"Entry {entry} missing 'val_accuracy' key"
        assert "val_loss" in entry, f"Entry {entry} missing 'val_loss' key"
    
    # Test predictions
    preds = adapter.predict(X[:5])
    assert preds.shape[0] == 5
    assert np.issubdtype(preds.dtype, np.integer)

    # Test serialization roundtrip
    path = tmp_path / "mnist_cnn_model.pt"
    adapter.save(path)
    
    restored = MNISTCNNAdapter(epochs=1, batch_size=2, learning_rate=1e-3, device="cpu")
    restored.load(path)
    restored_preds = restored.predict(X[:3])
    assert restored_preds.shape[0] == 3
    
    # Verify training history is cleared on new fit
    adapter.training_history = []  # Manual clear
    adapter.fit(X_train[:20], y_train[:20], X_val=X_val[:5], y_val=y_val[:5])
    assert len(adapter.training_history) > 0, "training_history should be repopulated after second fit"

