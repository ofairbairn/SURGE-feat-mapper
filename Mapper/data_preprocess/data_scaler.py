"""Robust feature scaling for the SURGE Mapper pipeline.

Fit :class:`DataScaler` on training data only, then use ``transform`` on every
other split or dataset.  Keeping fitting separate from transformation prevents
validation and test data from leaking into the preprocessing statistics.
"""

from __future__ import annotations

from typing import Tuple

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


__all__ = ["DataScaler"]
