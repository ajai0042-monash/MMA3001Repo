"""
interpolation.py: Reconstructs gaps using interpolation.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator


# Converts a datetime index to float seconds since the first point
def to_seconds(series):
    """
    Convert the datetime index to float seconds from the first point.

    Interpolation routines require numeric x values 
    note on independent-variable units).

    Parameters
    ----------
    series : pandas.Series
        Time-indexed series whose index will be converted.

    Returns
    -------
    numpy.ndarray
        1D array of floats, each entry is the number of seconds
        since `series.index[0]`.
    """
    return (series.index - series.index[0]).total_seconds().to_numpy()


# Fills gaps by drawing straight lines between the known points on each side
def reconstruct_linear(gapped_series):
    """
    Fill the gaps using linear interpolation.

    Uses `numpy.interp` to draw a straight line between the two known
    points on either side of each gap. Returns the series unchanged if
    there are fewer than two known points.

    Parameters
    ----------
    gapped_series : pandas.Series
        Time-indexed series containing NaN inside the gap region.

    Returns
    -------
    pandas.Series
        Same series with the NaN entries replaced by linear
        interpolation.
    """
    s = gapped_series.copy()
    known = s.notna()
    gap = s.isna()

    if known.sum() < 2:
        return s

    x = to_seconds(s)
    # Linear interpolation between known points
    s.iloc[gap.values] = np.interp(x[gap], x[known], s.values[known])
    return s


# Fills gaps using a smooth natural cubic spline (can overshoot on long gaps)
def reconstruct_cubic_spline(gapped_series):
    """
    Fill the gaps using a natural cubic spline.

    Natural boundary conditions set the second derivative to zero at
    the ends. Cubic splines can overshoot between knots on long gaps,
    which is a documented failure mode of this method.

    Parameters
    ----------
    gapped_series : pandas.Series
        Time-indexed series containing NaN inside the gap region.

    Returns
    -------
    pandas.Series
        Same series with the NaN entries replaced by cubic-spline
        interpolation. Falls back to linear interpolation if there
        are fewer than four known points.
    """
    s = gapped_series.copy()
    known = s.notna()
    gap = s.isna()

    if known.sum() < 4:
        return reconstruct_linear(gapped_series)

    x = to_seconds(s)
    #Construct the spline interpolant
    spline = CubicSpline(x[known], s.values[known], bc_type="natural")
    # Evaluate the spline at gap locations
    s.iloc[gap.values] = spline(x[gap])
    return s


# Fills gaps using PCHIP (shape-preserving cubic, no overshoot between monotonic points)
def reconstruct_pchip(gapped_series):
    """
    Fill the gaps using PCHIP interpolation.

    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is
    shape-preserving, so it will not overshoot between monotonic data
    points. This usually gives a safer reconstruction than a cubic
    spline on building-sensor data.

    Parameters
    ----------
    gapped_series : pandas.Series
        Time-indexed series containing NaN inside the gap region.

    Returns
    -------
    pandas.Series
        Same series with the NaN entries replaced by PCHIP
        interpolation. Returns the series unchanged if there are
        fewer than two known points.
    """
    s = gapped_series.copy()
    known = s.notna()
    gap = s.isna()

    if known.sum() < 2:
        return s

    x = to_seconds(s)
    #PchipInterpolator construction
    pchip = PchipInterpolator(x[known], s.values[known])
    # Evaluate at gap locations
    s.iloc[gap.values] = pchip(x[gap])
    return s
