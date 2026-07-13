"""Model adapters used by SURGE."""

from .mnist_cnn import MNISTCNNAdapter
from .owen_vae import OwenVAEAdapter

__all__ = ["MNISTCNNAdapter", "OwenVAEAdapter"]
