"""
integration.py: Cumulative degree-hours via the trapezoidal rule.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, trapezoid


# Integrates the temperature excess above a threshold to get degree-hours and the running total
def compute_degree_hours(series, threshold_c=26.0):
    """
    Integrate temperature excess above a threshold (deg C . h).

    Uses the trapezoidal rule on the real time coordinates (so uneven
    sampling is handled correctly) and only counts temperature above
    the threshold. Returns both the total exposure and the running
    cumulative curve.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed temperature series in degrees Celsius. NaN
        entries are dropped before integration.
    threshold_c : float, optional
        Temperature threshold in degrees Celsius above which exposure
        starts counting (default 26.0).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'total_degree_hours': float, total exposure in degC.h.
        - 'cumulative': pandas.Series, running integral at each
          timestamp.
        - 'max_excess_c': float, maximum excess above threshold.
        - 'hours_above_threshold': float, total time spent above
          the threshold in hours.
    """
    clean = series.dropna()#remove NaN before use
    if len(clean) < 2:
        return {
            "total_degree_hours": 0,
            "cumulative": pd.Series(dtype=float),
            "max_excess_c": 0,
            "hours_above_threshold": 0,
        }

    # Convert datetime index to hours
    t_h = (clean.index - clean.index[0]).total_seconds().to_numpy() / 3600

    # Clip to only count exceedance above the threshold
    excess = np.maximum(clean.values - threshold_c, 0.0)

    # total area under the excess curve
    total = trapezoid(excess, x=t_h)

    # running total at each timestamp
    cum = cumulative_trapezoid(excess, x=t_h, initial=0.0)

    dt = np.diff(t_h)
    hrs_above = np.sum(dt[excess[:-1] > 0])

    return {
        "total_degree_hours": total,
        "cumulative": pd.Series(cum, index=clean.index),
        "max_excess_c": float(np.max(excess)),
        "hours_above_threshold": hrs_above,
    }


# Builds a small table comparing degree-hours and max excess between the original and a reconstruction
def compare_degree_hours(original, reconstructed, threshold_c=26.0):
    """
    Compare degree-hours between original and reconstructed series.

    Calls `compute_degree_hours` on both series and formats the
    results into a 3-row table (original, reconstructed, difference)
    for easy inclusion in the notebook or report.

    Parameters
    ----------
    original : pandas.Series
        Ground-truth temperature series.
    reconstructed : pandas.Series
        Filled temperature series produced by one of the reconstruction
        methods.
    threshold_c : float, optional
        Temperature threshold in degrees Celsius above which exposure
        counts (default 26.0).

    Returns
    -------
    pandas.DataFrame
        Table with columns 'Series', 'Degree-hours [degC.h]', and
        'Max excess [degC]', and three rows: Original, Reconstructed,
        and Difference.
    """
    o = compute_degree_hours(original, threshold_c)
    r = compute_degree_hours(reconstructed, threshold_c)

    return pd.DataFrame({
        "Series": ["Original", "Reconstructed", "Difference"],
        "Degree-hours [degC.h]": [
            o["total_degree_hours"],
            r["total_degree_hours"],
            r["total_degree_hours"] - o["total_degree_hours"],
        ],
        "Max excess [degC]": [
            o["max_excess_c"],
            r["max_excess_c"],
            r["max_excess_c"] - o["max_excess_c"],
        ],
    })


# Plots cumulative degree-hour curves for several series on one chart for visual comparison
def plot_cumulative_exposure(series_dict, threshold_c=26.0, title=""):
    """
    Plot cumulative degree-hour curves.

    Draws one line per series showing how exposure builds up over
    time, with each line's total degree-hours shown in the legend.

    Parameters
    ----------
    series_dict : dict of {str: pandas.Series}
        Mapping of label to time-indexed temperature series. Each
        label becomes a line in the plot.
    threshold_c : float, optional
        Temperature threshold in degrees Celsius above which exposure
        counts (default 26.0).
    title : str, optional
        Title prefix shown above the chart; the threshold is appended
        automatically (default empty).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the cumulative exposure curves.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, s in series_dict.items():
        r = compute_degree_hours(s, threshold_c)
        if len(r["cumulative"]) > 0:
            ax.plot(
                r["cumulative"].index,
                r["cumulative"].values,
                label=f"{label} ({r['total_degree_hours']:.1f} degC.h)",
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative [degC.h]")
    ax.set_title(f"{title} (threshold = {threshold_c} degC)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
