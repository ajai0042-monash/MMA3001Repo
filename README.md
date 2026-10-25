# MMA3001 Project — Gap Reconstruction for Building Temperature Sensors

Project for MMA3001: Numerical Methods and Machine Learning, Monash University, 2026 S2.

Author: Abhinav Jain

## Engineering Problem

Environmental sensors in modern buildings produce long time series that feed into HVAC control, comfort scoring, and energy audits. In practice these series are never clean — network dropouts, firmware faults, and sensor downtime leave gaps that downstream analysis has to deal with. The usual fix is to fill them with linear interpolation and move on, but that choice is rarely tested against anything.

This project reconstructs artificial gaps in the Monash Smart Infrastructure temperature data using six methods (3 interpolation, 2 ML, 1 baseline), measures how each one scales with gap length, and asks whether the choice of fill actually changes a downstream engineering decision (degree-hours of thermal exposure).

Status: complete. All six methods implemented, validated against 54 artificial gaps (3 durations x 3 placements x 6 methods), checked for physical plausibility and engineering impact, and the methodology transfer-validated on pressure data. The written report is submitted via Moodle separately.

## Results

Best overall: **SVR (RBF)** - MAE 0.645 degC across all gap durations, 72.9% improvement over the mean baseline. But best-by-gap-length is more useful than best-overall:

| Gap size | Winner | MAE (degC) |
|---|---|---|
| 1h | Linear | 0.146 |
| 6h | PCHIP | 0.250 |
| 24h | SVR (RBF, windowed) | 0.934 |

### Overall ranking across all gap sizes

| Method | MAE (degC) | RMSE (degC) | R2 |
|---|---|---|---|
| SVR (RBF, windowed) | 0.645 | 0.751 | -2.607 |
| PCHIP | 0.692 | 0.864 | -0.009 |
| Linear Interp | 0.806 | 0.980 | 0.043 |
| MLP Neural Net (windowed) | 0.847 | 0.979 | -4.231 |
| Cubic Spline | 1.599 | 1.844 | -3.047 |
| Mean Baseline | 2.381 | 2.612 | -105.533 |

Negative R2 values appear on single-gap scenarios where the hidden ground truth is nearly constant (any method with modest absolute error can score below the "always predict the mean" baseline for that short window, because R2 is defined relative to the variance of the target). The overall MAE ranking is the honest story and tracks what you would feel if you looked at the reconstructions by eye.

### Key finding: method choice depends on gap length, not variable name

- Short gaps (<=6h): interpolation wins because the daily cycle isn't moving much. Linear and PCHIP both do fine.
- Long gaps (24h): windowed SVR wins by ~2x over any interpolation. The reason is visible in the residuals - all three interpolation methods leave a smooth 5-10 degC hump across the gap, which is exactly the diurnal cycle they cannot invent. SVR learned that cycle from the 48h training window and reconstructs it.
- Cubic Spline is the worst non-baseline method on 24h gaps (MAE 4.09 degC) because it overshoots dramatically - a known failure mode for natural splines on long extrapolations.

## Degree-Hours Engineering Impact

Does the choice of fill method actually change a comfort decision? Tested on the 24h gap zoomed to +/-6h, threshold 20 degC:

| Series | Degree-hours above 20 degC |
|---|---|
| Original (ground truth) | 48.8 degC.h |
| Linear / Cubic Spline / PCHIP | 4.4 degC.h |
| SVR (windowed) | 49.5 degC.h |
| MLP (windowed) | 64.9 degC.h |

Interpolation methods **underestimate thermal exposure by 91%** on a 24h gap because they fail to reconstruct the daytime temperature peak. SVR is within 1.5% of the true value. MLP overshoots by 33%. For any engineering decision tied to a threshold - comfort compliance, chiller sizing, energy-budget verification - this gap matters.

## Physical Plausibility (dT/dt)

All five methods pass the dT/dt < 2 degC/min indoor plausibility check at both 6h and 24h gaps (max |dT/dt| = 0.175 degC/min across the board). No method produces physically impossible rates of change - but passing this test does not mean the reconstruction is correct, as the degree-hours table above shows.

## Transfer Validation on Pressure

To check whether the methodology generalises beyond temperature, the same validation protocol was re-run on the barometric pressure channel from the same sensor:

| Method | MAE (mb) | RMSE (mb) | R2 |
|---|---|---|---|
| PCHIP | 0.524 | 0.642 | 0.436 |
| Linear | 0.564 | 0.656 | 0.428 |
| SVR (windowed) | 1.374 | 1.598 | -0.611 |

PCHIP dominates pressure. SVR performs much worse than on temperature. This is because pressure is weather-driven and smooth - there's no diurnal cycle for SVR to exploit, so the local window gives it nothing to learn and it adds noise. The lesson is that method choice depends on signal character, not variable name.

## Runtime and FLOPS

| Method | Mean runtime (24h gap) | FLOPS |
|---|---|---|
| Linear | 0.002 s | 288 |
| PCHIP | 0.003 s | 232,592 |
| Cubic Spline | 0.004 s | 232,592 |
| SVR (windowed) | 0.012 s | not reported |
| MLP (windowed) | 0.960 s | not reported |

FLOPS are only reported for the interpolation methods. SVR and MLP are not easy to express as a clean operation count because the SVR kernel evaluations depend on the number of support vectors chosen during fitting, and the MLP cost depends on how many optimiser iterations the Adam solver runs before early stopping. Wall-clock runtime is used instead for the ML methods.

Interpolation construction cost is dominated by `n_known` (~29k points), evaluation cost scales with `n_gaps` (6, 36, 144). Windowed SVR and MLP both train on ~576 points (+/-48h) regardless of gap size - this is the original contribution over a naive global ML fit, which took ~50s per gap before windowing. SVR is 100x faster than MLP for near-identical accuracy at 24h gaps.

## Data

Source: Monash Smart Infrastructure Environmental Data (unit-provided).

- 179,447 raw rows across 5 sensors (May 2023 - Apr 2026)
- 3 of 5 sensors (0241, 0227, 0777) have firmware timestamp corruption - their records collapse to a single datetime, making them unusable for time-series analysis
- Sensor **6012002000326** has 29,039 valid temperature readings from Dec 2023 to Apr 2026
- The dense usable period starts at **2025-09-01** (29,038 points at 10-minute median sampling)
- Range: 11.8 - 37.5 degC, mean 20.8 degC
- Raw CSV not committed (see `.gitignore`)

## Repository Structure

MMA3001Repo/

├── src/ # Python modules

│ ├── data_loader.py # CSV + JSON parsing, cleaning, audit

│ ├── gap_generator.py # Reproducible artificial gaps

│ ├── interpolation.py # Linear, Cubic Spline, PCHIP

│ ├── ml_models.py # SVR, MLP, DummyRegressor baseline

│ ├── evaluation.py # Metrics, residuals, plausibility

│ ├── integration.py # Trapezoidal degree-hours

│ └── profiling_utils.py # Runtime and FLOPS

├── MyProjectMMA3001.ipynb # Main analysis notebook

├── tests/ # pytest suite

├── doc/ # pdoc-generated HTML API docs

├── config/ # requirements.txt, project constants

├── tools/ # setup / test / docs shell scripts

├── res/ # Exported figures

├── README.md # This file

└── LICENSE # MIT

## Setup

```bash
git clone https://github.com/ajai0042-monash/MMA3001Repo.git
cd MMA3001Repo
bash tools/setup_env.sh
```

## Running the Analysis

Open `MyProjectMMA3001.ipynb` in Jupyter or Google Colab, mount Google Drive containing the dataset CSV, and run cells top-to-bottom.

## Tests and Documentation

```bash
bash tools/run_tests.sh
bash tools/generate_docs.sh
```

Latest test report: `tests/pytest_report.txt`. API docs: open `doc/index.html` in a browser.

## Limitations

- Only one sensor's data is usable (3 of 5 are corrupt). Results generalise to one building zone, not to the full installation.
- Dense period covers Sep 2025 - Apr 2026 only. Winter data with weaker diurnal amplitude might shift the interpolation-SVR crossover.
- SVR hyperparameters (C=10, epsilon=0.5) are reasonable defaults but not cross-validated. A proper grid search could improve 24h performance further.
- ML features are purely temporal (hour, weekday). Adding neighbouring-sensor readings or external weather would be the obvious next step but is out of scope for a numerical-methods unit.
- Degree-hours is reported for the 24h gap only. Impact on other engineering metrics (ventilation rate, chiller load) was not quantified.

## AI Use

Claude (Anthropic, Opus 4.7) and Google Colab AI features were used for code scaffolding, debugging, and documentation. All AI-generated content was reviewed, tested, and verified against the MMA3001 Weeks 1-7 course material. Full disclosure and reflection are in Section 7 of the project report.

## License

MIT - see LICENSE.
