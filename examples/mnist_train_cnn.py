#runner script for mnist_test_cnn.py
#pedagogical example for Owen
#mnist_test_cnn.py is located in SURGE/surge/model/mnist_test_cnn.py
import numpy as np
from surge.model import MNISTCNNAdapter

# Load data
x_train = np.load(r"C:\Users\Bipo1\Downloads\MNIST Data\mnist_train_features.npy")
y_train = np.load(r"C:\Users\Bipo1\Downloads\MNIST Data\mnist_train_targets.npy")

x_test = np.load(r"C:\Users\Bipo1\Downloads\MNIST Data\mnist_test_features.npy")
y_test = np.load(r"C:\Users\Bipo1\Downloads\MNIST Data\mnist_test_targets.npy")
# The adapter accepts flattened or image-shaped arrays.
# If your arrays are already shape (N, 784), use them directly.
# If they are shape (N, 28, 28), that also works.
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
model = MNISTCNNAdapter(
    epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    device="cpu",
)

model.fit(x_train, y_train)

# Predict on test data and report a simple classification metric
preds = model.predict(x_test)
y_test = np.asarray(y_test).reshape(-1)
preds = np.asarray(preds).reshape(-1)

accuracy = 100.0 * np.mean(preds == y_test)
print("First 10 predictions:", preds[:10])
print(f"Test accuracy: {accuracy:.2f}%")