"""Backend implementations used by SURGE model adapters."""

from .mnist_cnn import MNISTCNNModel
from .owen_vae import OwenVAEModel

__all__ = ["MNISTCNNModel", "OwenVAEModel"]
