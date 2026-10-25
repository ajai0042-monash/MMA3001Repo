"""
data_loader.py: Loads, cleans and audits the Monash Smart Infrastructure
environmental sensor CSV
"""

import json
import numpy as np
import pandas as pd

# Renaming dictionary of long JSON names to short column names
Sensor_column_map = {
    "Battery": "battery_v",
    "Carbon dioxide": "co2_ppm",
    "Humidity": "humidity_pct",
    "Pressure": "pressure_mb",
    "Temperature": "temperature_c",
    "Light Volatile Organic Compounds": "lvoc_ppb",
    "Particulate matter 2.5": "pm25_ug_m3",
}

# Physical plausibility bounds for each sensor variable (anything outside bounds is not considered)
sensor_bounds = {
    "temperature_c": (5.0,50.0),
    "co2_ppm": (350,5000),
    "humidity_pct": (0,100),
    "pressure_mb": (950,1050),
    "lvoc_ppb": (0,10000),
    "pm25_ug_m3": (0,1000),
}

# Unpacks one row’s nested JSON into a flat dict
def read_json_row(json_string):
    """
    Unpack a row of sensor JSON into a flat dictionary.

    Parameters
    ----------
    json_string : str
        Raw JSON string from the 'jsondata' column of the CSV.

    Returns
    -------
    dict
        Sensor readings with short column names (e.g. temperature_c) and
        a 'unix_timestamp' field. Returns an empty dict if the JSON cannot
        be parsed.
    """
    try:
        entries = json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return {}

    row = {}
    timestamp = None

    for entry in entries:
        sensor_name = entry["variable"]["name"]
        readings = entry["values"]
        if len(readings) == 0:
            continue

        value = readings[0].get("value")
        time_val = readings[0].get("time")

        if timestamp is None:
            timestamp = time_val

        short_name = Sensor_column_map.get(sensor_name)
        if short_name is not None and value is not None:
            row[short_name] = value

    row["unix_timestamp"] = timestamp
    return row

# Loads the whole CSV and cleans it
def load_env_csv(filepath, max_rows=None):
    """
    Load the environment CSV and flatten the nested JSON column.

    Parameters
    ----------
    filepath : str
        Path to the environmental sensor CSV on disk.
    max_rows : int, optional
        If given, only load the first `max_rows` rows (useful for quick
        testing). Loads the whole file by default.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with one column per sensor variable, a Melbourne
        local-time 'datetime' column, and a string 'sensorid' column.
    """
    #Read raw data into a DataFrame
    raw = pd.read_csv(filepath, nrows=max_rows)
    print(f"Loaded {len(raw)} rows")

    # Apply read_json_row to every row
    parsed = raw["jsondata"].apply(read_json_row)
    flat = pd.DataFrame(parsed.tolist())

    data = raw.drop(columns=["jsondata"])
    data = pd.concat([data.reset_index(drop=True),
                      flat.reset_index(drop=True)], axis=1)

    # Convert Unix seconds to Melbourne local datetime
    data["datetime"] = pd.to_datetime(
        data["unix_timestamp"], unit="s", utc=True, errors="coerce"
    ).dt.tz_convert("Australia/Melbourne")

    # Clean sensorid from scientific notation to string
    data["sensorid"] = data["sensorid"].astype(float).astype(int).astype(str)

    print(f"Parsed into {len(flat.columns)} sensor columns")
    return data


# Filters down to one sensor and one variable
def get_sensor_series(data, variable, sensor_id):
    """
    Extract a cleaned time series for one sensor and one variable.

    Audit-before-modelling pattern:
    schema check, missingness, duplicates, implausible observations.

    Parameters
    ----------
    data : pandas.DataFrame
        Cleaned DataFrame returned by `load_env_csv`.
    variable : str
        Short column name of the variable to extract (e.g. 'temperature_c').
    sensor_id : str or int
        ID of the sensor to filter to (will be cast to string).

    Returns
    -------
    pandas.Series
        Time-indexed series of the chosen variable for the chosen sensor,
        with NaNs, duplicates and implausible readings removed.
    """
    # Filter to the chosen sensor
    rows = data[data["sensorid"] == str(sensor_id)]
    print(f"Sensor {sensor_id}: {len(rows)} rows")

    # Drop rows where datetime or variable is NaN
    rows = rows.dropna(subset=["datetime", variable])

    # Remove physically implausible readings
    lo, hi = sensor_bounds[variable]
    ok = (rows[variable] >= lo) & (rows[variable] <= hi)
    bad = (~ok).sum()
    if bad > 0:
        print(f"Removed {bad} implausible {variable} readings")
    rows = rows[ok]

    # Set datetime index, remove duplicates and sort
    rows = rows.set_index("datetime")
    rows = rows[~rows.index.duplicated()].sort_index()

    series = rows[variable]
    print(f"{variable}: {len(series)} pts, {series.index[0]} to {series.index[-1]}")
    return series

# Extract temperature data of the sensor
def get_temperature_series(data, sensor_id):
    """
    Shortcut for temperature extraction.

    Parameters
    ----------
    data : pandas.DataFrame
        Cleaned DataFrame returned by `load_env_csv`.
    sensor_id : str or int
        ID of the sensor to filter to.

    Returns
    -------
    pandas.Series
        Time-indexed temperature series in degrees Celsius for the chosen
        sensor.
    """
    return get_sensor_series(data, "temperature_c", sensor_id)

# Prints the data health check
def audit_data(data):
    """
    Print a data quality audit.

    Audit_summary structure: row count,
    duplicates, missing values, validity rules.

    Parameters
    ----------
    data : pandas.DataFrame
        Cleaned DataFrame returned by `load_env_csv`.

    Returns
    -------
    None
        Prints the audit summary to stdout; does not return a value.
    """
    print(f"Total rows: {len(data)}")

    cols = [c for c in Sensor_column_map.values() if c in data.columns]

    # Missing value audit
    print("\nMissing Values")
    for c in cols:
        n = data[c].isna().sum()
        if n > 0:
            print(f" {c}: {n} ({100*n/len(data):.1f}%)")

    # Unique asset count
    print(f"\nUnique sensors: {data['sensorid'].nunique()}")

    # Value range audit
    print("\nValue Ranges")
    for c in cols:
        if data[c].notna().any():
            print(f"{c}: [{data[c].min():.2f}, {data[c].max():.2f}]")

    print(f"\nTime: {data['datetime'].min()} to {data['datetime'].max()}")


# Makes fake test data if the real CSV isn’t there
def generate_synthetic_data(start="2024-06-01", periods=2016, freq="10min",
                            noise_std=0.3, random_seed=42):
    """
    Synthetic indoor temperature series for testing without the real CSV.

    Builds a sine-wave diurnal signal plus slow random drift plus Gaussian
    noise so the pipeline can be tested end-to-end when the real dataset
    is not available.

    Parameters
    ----------
    start : str, optional
        Start timestamp for the synthetic series (default "2024-06-01").
    periods : int, optional
        Number of samples to generate (default 2016, which is 14 days at
        10-minute spacing).
    freq : str, optional
        Pandas frequency string for the sampling interval (default "10min").
    noise_std : float, optional
        Standard deviation of the additive Gaussian measurement noise in
        degrees Celsius (default 0.3).
    random_seed : int, optional
        Seed for the random number generator so results are reproducible
        (default 42).

    Returns
    -------
    pandas.Series
        Time-indexed synthetic temperature series in degrees Celsius.
    """
    #Seeded random number generator for reproducibility
    rng = np.random.default_rng(random_seed)

    # Build evenly spaced timestamps
    t = pd.date_range(start=start, periods=periods, freq=freq)

    # Hour-of-day fraction for the diurnal component
    hour = t.hour + t.minute / 60.0

    #Periodic signal — sine wave with afternoon peak
    diurnal = 3.0 * np.sin(2 * np.pi * (hour - 8) / 24)

    #Random drift as a slow trend
    drift = np.cumsum(rng.normal(0, 0.008, periods))

    #Measurement noise added to the true signal
    noise = rng.normal(0, noise_std, periods)
    temp = 22.0 + diurnal + drift + noise
    return pd.Series(temp, index=t, name="temperature_c")
