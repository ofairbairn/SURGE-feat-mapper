"""Robust feature scaling for the SURGE Mapper pipeline.

Fit :class:`DataScaler` on training data only, then use ``transform`` on every
other split or dataset.  Keeping fitting separate from transformation prevents
validation and test data from leaking into the preprocessing statistics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler


class DataScaler(RobustScaler):
    """Scale Mapper features using statistics that are robust to outliers.

    This is a Mapper-specific name for scikit-learn's ``RobustScaler``.  By
    default, each feature is centered on its training-set median and scaled by
    its interquartile range.  It implements the standard scikit-learn
    ``fit``/``transform`` API, so it can also be used in a pipeline.

    Parameters
    ----------
    with_centering:
        Center each feature on its median.
    with_scaling:
        Scale each feature using its quantile range.
    quantile_range:
        Quantiles used to calculate the scale. Values must satisfy
        ``0 < q_min < q_max < 100``.
    copy:
        Attempt to avoid in-place changes to the input data.
    unit_variance:
        Rescale the quantile range so normally distributed features have unit
        variance.

    Notes
    -----
    Call ``fit`` or ``fit_transform`` only with training data.  Calling
    ``transform`` does not update the fitted center or scale.
    """

    def __init__(
        self,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        copy: bool = True,
        unit_variance: bool = False,
    ) -> None:
        super().__init__(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            copy=copy,
            unit_variance=unit_variance,
        )


class ImageDataScaler:
    """Scale flattened image intensities without centering them below zero.

    Common ``[0, 1]`` inputs are left unchanged and common byte images are
    divided by 255. Other numeric ranges use a global train-only min/max
    transform. Validation and test values are clipped to the decoder-friendly
    ``[0, 1]`` interval.
    """

    def __init__(self, *, clip: bool = True) -> None:
        self.clip = bool(clip)
        self.data_min_: Optional[float] = None
        self.data_max_: Optional[float] = None
        self.offset_: Optional[float] = None
        self.scale_: Optional[float] = None
        self.method_: Optional[str] = None

    def fit(self, X, y=None) -> "ImageDataScaler":
        del y
        array = np.asarray(X, dtype=np.float32)
        if array.size == 0:
            raise ValueError("Cannot fit ImageDataScaler on an empty array")
        if not np.isfinite(array).all():
            raise ValueError("ImageDataScaler requires finite training values")
        self.data_min_ = float(np.min(array))
        self.data_max_ = float(np.max(array))
        tolerance = 1e-6
        if self.data_min_ >= -tolerance and self.data_max_ <= 1.0 + tolerance:
            self.offset_ = 0.0
            self.scale_ = 1.0
            self.method_ = "identity_0_1"
        elif self.data_min_ >= -tolerance and self.data_max_ <= 255.0 + tolerance:
            self.offset_ = 0.0
            self.scale_ = 255.0
            self.method_ = "divide_255"
        else:
            self.offset_ = self.data_min_
            value_range = self.data_max_ - self.data_min_
            self.scale_ = value_range if value_range > 0.0 else 1.0
            self.method_ = "global_train_minmax"
        return self

    def _check_fitted(self) -> Tuple[float, float]:
        if self.offset_ is None or self.scale_ is None:
            raise ValueError("ImageDataScaler must be fitted before transformation")
        return self.offset_, self.scale_

    def transform(self, X) -> np.ndarray:
        offset, scale = self._check_fitted()
        transformed = (np.asarray(X, dtype=np.float32) - offset) / scale
        if self.clip:
            transformed = np.clip(transformed, 0.0, 1.0)
        return transformed.astype(np.float32, copy=False)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        offset, scale = self._check_fitted()
        return (np.asarray(X, dtype=np.float32) * scale + offset).astype(
            np.float32,
            copy=False,
        )


__all__ = ["DataScaler", "ImageDataScaler"]
