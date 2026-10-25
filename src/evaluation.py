"""
evaluation.py:Metrics, residuals and method comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Calculates MAE, RMSE and R2 for one set of actual vs predicted values
def compute_metrics(actual, predicted):
    """
    Return MAE, RMSE and R2 between actual and predicted.

    Thin wrapper around scikit-learn metrics. RMSE is computed as the
    square root of mean squared error. All three metrics are stored
    in a dictionary so they can be unpacked straight into a results row.

    Parameters
    ----------
    actual : array-like
        Ground-truth values.
    predicted : array-like
        Model-predicted values, same length as `actual`.

    Returns
    -------
    dict
        Dictionary with keys 'MAE', 'RMSE', 'R2'.
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


# Runs every reconstruction method on every gap scenario and collects the metrics into one table
def run_full_comparison(scenarios, reconstruct_functions):
    """
    Evaluate every method on every gap scenario and return a DataFrame.

    Loops over all scenarios and all methods, catches per-method
    failures so one bad method does not break the whole run, and
    returns a long-format table ready for groupby or plotting.

    Parameters
    ----------
    scenarios : list of dict
        Output of `create_validation_gaps`. Each dict must contain
        'gapped_series', 'ground_truth', 'duration_hours' and 'gap_id'.
    reconstruct_functions : dict of {str: callable}
        Mapping of method name to a function that takes a gapped
        series and returns a filled series.

    Returns
    -------
    pandas.DataFrame
        One row per (method, scenario) with columns 'Method',
        'duration_hours', 'gap_id', 'MAE', 'RMSE', 'R2'. NaN appears
        in the metric columns if a method raised an exception.
    """
    rows = []
    for sc in scenarios:
        g, t = sc["gapped_series"], sc["ground_truth"]
        for name, fn in reconstruct_functions.items():
            try:
                filled = fn(g)
                pred = filled.reindex(t.index)
                valid = pred.notna() & t.notna()
                if valid.sum() > 0:
                    m = compute_metrics(t[valid], pred[valid])
                else:
                    m = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
            except Exception as e:
                print(f"  {name} failed: {e}")
                m = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

            rows.append({
                "Method": name,
                "duration_hours": sc["duration_hours"],
                "gap_id": sc["gap_id"],
                **m,
            })
    return pd.DataFrame(rows)


# Draws a scatter plot of residuals (observed minus predicted) to look for structure
def plot_residuals(actual, predicted, method_name="", ax=None):
    """
    Residual scatter plot.

    A useful residual plot should not show a strong curve, drift, or
    funnel shape. Any structure means the method systematically
    misses a feature of the signal (for this project the diurnal
    cycle).

    Parameters
    ----------
    actual : array-like
        Ground-truth values.
    predicted : array-like
        Model-predicted values, same length as `actual`.
    method_name : str, optional
        Short label shown in the plot title (default empty).
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the residuals were drawn on.
    """
    residuals = np.asarray(actual) - np.asarray(predicted)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    ax.scatter(range(len(residuals)), residuals, s=12, alpha=0.6)
    ax.axhline(0.0, color="black", linestyle="--")
    ax.set_xlabel("Point index")
    ax.set_ylabel("Residual: observed - predicted")
    ax.set_title(f"Residuals: {method_name}")
    ax.grid(True, alpha=0.3)
    return ax


# Draws a predicted vs actual scatter with a y = x reference line
def plot_prediction_error(actual, predicted, method_name="", ax=None):
    """
    Predicted vs actual scatter with y = x reference line.

    Points close to the diagonal mean good predictions. Systematic
    offsets from the diagonal reveal bias.

    Parameters
    ----------
    actual : array-like
        Ground-truth values (x-axis).
    predicted : array-like
        Model-predicted values (y-axis), same length as `actual`.
    method_name : str, optional
        Short label shown in the plot title (default empty).
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. A new square figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the scatter was drawn on.
    """
    a = np.asarray(actual)
    p = np.asarray(predicted)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(a, p, s=12, alpha=0.6)
    bnd = [min(a.min(), p.min()), max(a.max(), p.max())]
    ax.plot(bnd, bnd, "k--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(method_name)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


# Draws a labelled bar chart comparing methods on a single error metric
def plot_comparison_bars(df, metric="MAE_C", title=""):
    """
    Bar chart comparing methods on a single metric.

    Draws one bar per method and labels each with its value. Used to
    show the overall ranking at a glance.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with a 'Method' column and a numeric column named
        `metric`.
    metric : str, optional
        Column name to plot (default "MAE_C").
    title : str, optional
        Title shown above the chart (default empty).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(df["Method"], df[metric],
                  color="steelblue", edgecolor="black")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


# Checks whether a filled series ever changes faster than real indoor temperature could
def check_physical_plausibility(series, max_rate_c_per_min=2.0):
    """
    dT/dt plausibility check using numpy.gradient.

    Estimates the rate of change at every point using a centred
    finite difference with the real time coordinates, then counts
    how many values exceed a physically reasonable threshold.
    Indoor temperature rarely changes faster than ~2 degC/min, so
    anything above that suggests the reconstruction produced an
    unphysical spike.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed temperature series in degrees Celsius.
    max_rate_c_per_min : float, optional
        Threshold above which a rate of change is counted as a
        violation (default 2.0 degC/min).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'max_rate_c_per_min' : largest absolute dT/dt observed.
        - 'n_violations' : count of points above the threshold.
        - 'violation_fraction' : fraction of points above threshold.
        - 'plausible' : True if there were no violations.
    """
    # Convert the datetime index to minutes
    t_min = (series.index - series.index[0]).total_seconds().to_numpy() / 60

    #Centred finite-difference estimate with real coordinates
    dydt = np.gradient(series.values, t_min)

    max_rate = float(np.max(np.abs(dydt)))
    n_violations = int((np.abs(dydt) > max_rate_c_per_min).sum())

    return {
        "max_rate_c_per_min": max_rate,
        "n_violations": n_violations,
        "violation_fraction": n_violations / len(series),
        "plausible": bool(n_violations == 0),
    }
