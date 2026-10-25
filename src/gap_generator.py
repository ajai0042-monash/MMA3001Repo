"""
gap_generator.py: Creates artificial gaps for controlled validation

Hold backknown values so error can be measured, and the split must match the deployment question.
"""

import numpy as np
import pandas as pd

# Finds unbroken stretches of valid (no NaN) data in the series
def find_complete_segments(series, min_length=100):
    """
    Find contiguous stretches with no NaN values.

    Scans the series once and groups neighbouring valid points into
    (start_index, end_index) ranges. Only ranges longer than
    `min_length` are kept, so very short islands of valid data between
    NaN runs are ignored.

    Parameters
    ----------
    series : pandas.Series
        Time-indexed input series that may contain NaN values.
    min_length : int, optional
        Minimum number of consecutive valid points for a stretch to
        count as a usable segment (default 100).

    Returns
    -------
    list of tuple of int
        List of (start_index, end_index) pairs, each giving the row
        positions of one complete segment. Empty list if no segment
        meets `min_length`.
    """
    valid = series.notna().to_numpy()
    segments = []
    start = None

    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_length:
                segments.append((start, i - 1))
            start = None

    # Catch a stretch that runs to the end
    if start is not None and len(valid) - start >= min_length:
        segments.append((start, len(valid) - 1))

    return segments

# Hides a block of known values and returns both the gapped series and the hidden truth
def create_single_gap(series, gap_centre, gap_length):
    """
    Mask a block of known values, storing ground truth.

    The hidden values become the validation reference for later
    error calculation. The function always keeps at least one
    known point before and after the gap so interpolation methods
    have endpoints to work with.

    Parameters
    ----------
    series : pandas.Series
        Complete time-indexed series with no NaNs in the region
        that will be masked.
    gap_centre : int
        Row position (not timestamp) of the point the gap should
        be centred on.
    gap_length : int
        Number of consecutive samples to hide. Must be less than
        the total series length minus one.

    Returns
    -------
    gapped : pandas.Series
        Copy of the input series with NaN in the masked region.
    truth : pandas.Series
        The hidden true values, indexed by the same timestamps as
        the masked region. Used later to score reconstructions.

    Raises
    ------
    ValueError
        If `gap_length` is greater than or equal to the series
        length minus one.
    """
    n = len(series)

    if gap_length >= n - 1:
        raise ValueError(
            f"Gap length {gap_length} too large for {n}-point series."
        )

    half = gap_length // 2
    gap_start = max(gap_centre - half, 1) # keep at least one point before
    gap_end = min(gap_start + gap_length, n - 1) # keep at least one point after

    # Save the true values before hiding them (ground truth for validation)
    truth = series.iloc[gap_start:gap_end].copy()

    # Create the gapped version with NaN in the masked region
    gapped = series.copy()
    gapped.iloc[gap_start:gap_end] = np.nan

    return gapped, truth

# Creates multiple artificial gap scenarios (several durations, several positions) for validation
def create_validation_gaps(series, gap_durations_hours=(1, 6, 24),
                           n_gaps_per_duration=3, random_seed=42):
    """
    Generate multiple gap scenarios for evaluation.

    - Seeded RNG so the same gaps are produced every run.
    - Gaps are placed inside the longest complete stretch of the
      series so no gap accidentally overlaps a real missing value.
    - Multiple scenarios per duration so the final comparison is
      not swayed by one unlucky gap position.

    Parameters
    ----------
    series : pandas.Series
        Complete time-indexed input series (one sensor, one variable).
    gap_durations_hours : tuple of int, optional
        Lengths of the artificial gaps to create, in hours
        (default (1, 6, 24)).
    n_gaps_per_duration : int, optional
        How many random positions to try per gap duration
        (default 3).
    random_seed : int, optional
        Seed for the random number generator so the gap positions
        are reproducible (default 42).

    Returns
    -------
    list of dict
        One dict per scenario, with keys:
        - 'duration_hours': int, the gap length in hours.
        - 'gap_id': int, 0-based index within the duration group.
        - 'gapped_series': pandas.Series with NaN inside the gap.
        - 'ground_truth': pandas.Series of the hidden true values.

    Raises
    ------
    ValueError
        If the input series contains no complete segment of at least
        10 points.
    """
    # Seeded random number generator for reproducibility
    rng = np.random.default_rng(random_seed)

    # Median sampling interval
    diffs = np.diff(series.index.asi8) / 1e9  # seconds between samples
    dt_s = np.median(diffs)
    samples_per_hour = 3600 / dt_s
    print(f"Median sampling interval: {dt_s:.1f}s ({dt_s/60:.1f} min)")

    # Place gaps inside the longest complete stretch
    segments = find_complete_segments(series, min_length=10)
    if not segments:
        raise ValueError("No complete segments found.")
    s_start, s_end = max(segments, key=lambda s: s[1] - s[0])
    segment = series.iloc[s_start:s_end + 1].copy()

    scenarios = []
    for dur in gap_durations_hours:
        # Convert duration in hours to a number of samples
        gap_length = int(dur * samples_per_hour)
        if gap_length >= len(segment) - 2:
            print(f"Warning: {dur}h gap too large, skipped.")
            continue

        # Keep a margin so the gap does not touch segment edges
        margin = gap_length // 2 + 1
        possible_centres = np.arange(margin, len(segment) - margin)

        # W5.6 §5.6.6 — seeded random selection of split locations
        chosen = rng.choice(
            possible_centres,
            size=min(n_gaps_per_duration, len(possible_centres)),
            replace=False)

        for gap_id, centre in enumerate(sorted(chosen)):
            gapped, truth = create_single_gap(segment, centre, gap_length)
            scenarios.append({
                "duration_hours": dur,
                "gap_id": gap_id,
                "gapped_series": gapped,
                "ground_truth": truth,
            })

    print(f"Generated {len(scenarios)} gap scenarios.")
    return scenarios
