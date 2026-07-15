"""Model adapters used by SURGE."""

from .autoencoder import AutoencoderAdapter
from .mnist_cnn import MNISTCNNAdapter
from .owen_vae import OwenVAEAdapter

__all__ = ["AutoencoderAdapter", "MNISTCNNAdapter", "OwenVAEAdapter"]
