"""Compatibility shim for the legacy MNIST CNN adapter module."""

from __future__ import annotations

from .adapters.mnist_cnn import MNISTCNNAdapter

__all__ = ["MNISTCNNAdapter"]