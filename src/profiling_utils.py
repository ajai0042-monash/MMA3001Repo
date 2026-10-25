"""
profiling_utils.py: Timing and performance tools
"""

import time
import numpy as np
import pandas as pd


# Runs a function several times and reports the min, mean and max wall-clock time
def time_function(func, *args, n_repeats=3, **kwargs):
    """
    Time a callable over n_repeats runs and return min/mean/max seconds.

    Uses `time.perf_counter` which is the highest
    resolution clock available on the platform. Returns the result of
    the last run alongside the timings so the caller can use both.

    Parameters
    ----------
    func : callable
        Function to time.
    *args
        Positional arguments passed straight to `func`.
    n_repeats : int, optional
        Number of times to run `func` (default 3). More repeats give
        a more stable mean but take longer.
    **kwargs
        Keyword arguments passed straight to `func`.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'min_s' : fastest single run in seconds.
        - 'mean_s' : average run time in seconds.
        - 'max_s' : slowest single run in seconds.
        - 'result' : value returned by the final call to `func`.
    """
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter() # start the timer
        result = func(*args, **kwargs)  # run the block being measured
        times.append(time.perf_counter() - t0) # record elapsed seconds

    return {
        "min_s": min(times),
        "mean_s": np.mean(times),
        "max_s": max(times),
        "result": result,
    }


# Times every reconstruction method on the same gapped input and returns a comparison table
def compare_runtimes(methods, gapped_series, n_repeats=3):
    """
    Time multiple reconstruction methods on the same input.

    Loops over the given method dictionary, calls `time_function` on
    each one with the same gapped series, and collects the timings
    into a DataFrame for easy sorting and plotting.

    Parameters
    ----------
    methods : dict of {str: callable}
        Mapping of method name to a function that takes a gapped
        series and returns a filled series.
    gapped_series : pandas.Series
        Time-indexed series containing NaN inside the gap region,
        passed to every method.
    n_repeats : int, optional
        Number of times to run each method (default 3).

    Returns
    -------
    pandas.DataFrame
        One row per method with columns 'Method', 'min_s', 'mean_s',
        'max_s'.
    """
    rows = []
    for name, fn in methods.items():
        t = time_function(fn, gapped_series, n_repeats=n_repeats)
        rows.append({
            "Method": name,
            "min_s": t["min_s"],
            "mean_s": t["mean_s"],
            "max_s": t["max_s"],
        })
    return pd.DataFrame(rows)


# Estimates the floating-point operations needed for an interpolation method
def estimate_interpolation_flops(n_known, n_gaps, method="pchip"):
    """
    Rough FLOPS estimate for an interpolation method.

    Linear interpolation has no construction cost and roughly two
    operations per gap point. Spline and PCHIP both need a
    tridiagonal solve across the known knots (approximately 8 FLOPs
    per known point) plus a cubic evaluation per gap point
    (approximately 10 FLOPs).

    Parameters
    ----------
    n_known : int
        Number of known (non-NaN) points in the series.
    n_gaps : int
        Number of gap points that need to be filled.
    method : {"linear", "pchip", "cubic_spline"}, optional
        Which interpolation method's FLOP cost to estimate
        (default "pchip"). Anything other than "linear" uses the
        spline/PCHIP formula.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'construction_flops' : cost of building the interpolant.
        - 'evaluation_flops' : cost of evaluating at the gap points.
        - 'total_flops' : sum of the two.
    """
    if method == "linear":
        return {
            "construction_flops": 0,
            "evaluation_flops": 2 * n_gaps,
            "total_flops": 2 * n_gaps,
        }

    # Spline/PCHIP: tridiagonal solve across known knots + cubic evaluation
    return {
        "construction_flops": 8 * n_known,
        "evaluation_flops": 10 * n_gaps,
        "total_flops": 8 * n_known + 10 * n_gaps,
    }
