"""Unit tests for the MMA3001 project modules. Run with: pytest tests/ -v"""
import sys, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from interpolation import reconstruct_linear, reconstruct_cubic_spline, reconstruct_pchip
from gap_generator import create_single_gap, find_complete_segments
from evaluation import compute_metrics, check_physical_plausibility
from integration import compute_degree_hours
from data_loader import generate_synthetic_data
from profiling_utils import time_function, estimate_interpolation_flops


def test_synthetic_data_shape():
    s = generate_synthetic_data(periods=100)
    assert len(s) == 100 and s.notna().all()

def test_single_gap_creation():
    s = generate_synthetic_data(periods=100)
    gapped, truth = create_single_gap(s, gap_centre=50, gap_length=10)
    assert gapped.isna().sum() == 10 and len(truth) == 10

def test_linear_fills_all_gaps():
    s = generate_synthetic_data(periods=100)
    gapped, _ = create_single_gap(s, gap_centre=50, gap_length=10)
    assert reconstruct_linear(gapped).isna().sum() == 0

def test_pchip_matches_endpoints():
    s = generate_synthetic_data(periods=100)
    gapped, _ = create_single_gap(s, gap_centre=50, gap_length=10)
    filled = reconstruct_pchip(gapped)
    assert filled.iloc[0] == s.iloc[0] and filled.iloc[-1] == s.iloc[-1]

def test_cubic_spline_fills_all_gaps():
    s = generate_synthetic_data(periods=100)
    gapped, _ = create_single_gap(s, gap_centre=50, gap_length=10)
    assert reconstruct_cubic_spline(gapped).isna().sum() == 0

def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(y, y)
    assert m["MAE"] == 0.0 and m["R2"] == 1.0

def test_metrics_mae_calculation():
    m = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]))
    assert m["MAE"] == 0.5

def test_degree_hours_below_threshold():
    idx = pd.date_range("2025-01-01", periods=50, freq="10min")
    s = pd.Series([18.0] * 50, index=idx)
    assert compute_degree_hours(s, threshold_c=26.0)["total_degree_hours"] == 0

def test_time_function_returns_timing():
    r = time_function(lambda: 42, n_repeats=2)
    assert r["result"] == 42

def test_flops_estimate_scales_with_input():
    assert estimate_interpolation_flops(200, 10, "pchip")["total_flops"] > estimate_interpolation_flops(100, 10, "pchip")["total_flops"]
