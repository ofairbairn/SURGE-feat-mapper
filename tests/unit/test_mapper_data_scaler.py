"""Tests for the Mapper's robust feature scaler."""

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler

from Mapper.preprocess.data_scaler import DataScaler, ImageDataScaler


def test_data_scaler_uses_median_and_interquartile_range() -> None:
    data = np.array([[1.0, 2.0], [1.5, 2.5], [1.2, 1.8], [100.0, 200.0]])

    scaler = DataScaler()
    scaled = scaler.fit_transform(data)

    np.testing.assert_allclose(scaler.center_, np.median(data, axis=0))
    np.testing.assert_allclose(
        scaler.scale_,
        np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0),
    )
    np.testing.assert_allclose(np.median(scaled, axis=0), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(scaler.inverse_transform(scaled), data)


def test_transform_does_not_refit_on_mapper_data() -> None:
    training_data = np.array([[1.0, 2.0], [1.5, 2.5], [1.2, 1.8]])
    mapper_data = np.array([[100.0, 200.0]])
    scaler = DataScaler().fit(training_data)
    training_center = scaler.center_.copy()
    training_scale = scaler.scale_.copy()

    transformed = scaler.transform(mapper_data)

    np.testing.assert_allclose(scaler.center_, training_center)
    np.testing.assert_allclose(scaler.scale_, training_scale)
    np.testing.assert_allclose(
        transformed, (mapper_data - training_center) / training_scale
    )


def test_image_data_scaler_preserves_nonnegative_pixel_range() -> None:
    training = np.array([[0.0, 64.0], [128.0, 255.0]], dtype=np.float32)
    scaler = ImageDataScaler()

    transformed = scaler.fit_transform(training)

    assert scaler.method_ == "divide_255"
    assert transformed.dtype == np.float32
    assert float(transformed.min()) == 0.0
    assert float(transformed.max()) == 1.0
    np.testing.assert_allclose(scaler.inverse_transform(transformed), training)
    np.testing.assert_allclose(
        scaler.transform(np.array([[-5.0, 300.0]], dtype=np.float32)),
        np.array([[0.0, 1.0]], dtype=np.float32),
    )


def test_data_scaler_is_sklearn_pipeline_compatible() -> None:
    scaler = DataScaler(quantile_range=(10.0, 90.0), unit_variance=True)

    cloned = clone(scaler)

    assert isinstance(cloned, RobustScaler)
    assert cloned.quantile_range == (10.0, 90.0)
    assert cloned.unit_variance is True
