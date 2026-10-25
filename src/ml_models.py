"""
ml_models.py: ML reconstruction using SVR, MLP and a mean baseline.
Support Vector Regression, Neural Network Regression, DummyRegressor baseline
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# Builds hour-of-day and day-of-week features using sine/cosine encoding so the cycle wraps correctly
def build_temporal_features(dt_index):
    """
    Create cyclic temporal features from a DatetimeIndex.

    Hour-of-day and day-of-week are both cyclic, meaning hour 23 is
    close to hour 0 and Sunday is close to Monday. A raw integer
    encoding would break this. Sine/cosine encoding gives the model
    four smooth features that respect the wrap-around.

    Parameters
    ----------
    dt_index : pandas.DatetimeIndex
        Timestamps for every point the model needs features for
        (both training and gap points).

    Returns
    -------
    numpy.ndarray
        2D array of shape (n_points, 4) with columns
        [sin(hour), cos(hour), sin(weekday), cos(weekday)].
    """
    hour = dt_index.hour + dt_index.minute / 60.0
    weekday = dt_index.dayofweek

    # sine/cosine encoding so cyclic quantities wrap correctly
    X = np.column_stack([
        np.sin(2*np.pi*hour/24),
        np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*weekday/7),
        np.cos(2*np.pi*weekday/7),
    ])
    return X


# Fills gaps using one of SVR, MLP or a mean baseline, trained only on the known points
def reconstruct_ml(gapped_series, method="svr"):
    """
    Full ML reconstruction.

    Builds temporal features for every timestamp, trains the chosen
    model on the known points, then predicts the hidden gap points.
    StandardScaler is fitted on training data only (inside the
    Pipeline) and then applied consistently to the gap-location
    features at predict time.

    Parameters
    ----------
    gapped_series : pandas.Series
        Time-indexed series containing NaN inside the gap region.
    method : {"svr", "mlp", "baseline"}, optional
        Which model to use (default "svr").
        - "svr": Support Vector Regression with an RBF kernel
          (C=10, epsilon=0.5, gamma="scale").
        - "mlp": Multi-layer perceptron with hidden layers
          (64, 32), ReLU activation, Adam optimiser, early stopping.
        - "baseline": DummyRegressor that always predicts the mean.

    Returns
    -------
    pandas.Series
        Same series with the NaN entries replaced by model
        predictions. Returns the series unchanged if there are no
        gaps to fill.

    Raises
    ------
    ValueError
        If `method` is not one of the three supported options.
    """
    s = gapped_series.copy()
    known = s.notna() # points we can train on
    gap = s.isna() # points we need to predict

    if gap.sum() == 0:
        return s

    # build the X matrix for every timestamp
    X = build_temporal_features(s.index)
    X_train = X[known]
    y_train = s.values[known]
    X_gap = X[gap]

    # SVR with RBF kernel inside a Pipeline
    if method == "svr":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.5, gamma="scale")),
        ])

    # MLPRegressor with early stopping
    elif method == "mlp":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                random_state=42,
            )),
        ])

    #Mean baseline
    elif method == "baseline":
        model = DummyRegressor(strategy="mean")

    else:
        raise ValueError(f"Unknown method '{method}'")

    # Standard scikit-learn fit/predict pattern
    model.fit(X_train, y_train)
    s.iloc[gap.values] = model.predict(X_gap)
    return s
