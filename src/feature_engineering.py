import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GridFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline for energy grid anomaly detection.

    Computes rolling statistics, rate-of-change features, cross-sensor
    correlations, cyclical time encodings, and generation-load ratios.
    Compatible with scikit-learn Pipeline API.

    Parameters
    ----------
    rolling_windows : list of int or None
        List of window sizes for rolling statistics.
    include_cross_correlation : bool
        Whether to compute cross-sensor features.
    """

    def __init__(self, rolling_windows=None, include_cross_correlation=True):
        self.rolling_windows = rolling_windows or [5, 15, 30]
        self.include_cross_correlation = include_cross_correlation
        self._feature_names = None
        self._fitted = False

    def fit(self, X, y=None):
        """Fit the feature engineer (learns column names).

        Parameters
        ----------
        X : pandas.DataFrame
            Feature DataFrame with sensor readings.
        y : ignored
            Not used, present for API compatibility.

        Returns
        -------
        GridFeatureEngineer
            self
        """
        self._numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._sensor_cols = [c for c in self._numeric_cols
                            if any(c.startswith(p) for p in
                                   ["solar_", "wind_", "battery_", "hydrogen_", "grid_"])]
        self._fitted = True
        print(f"GridFeatureEngineer fitted on {len(self._sensor_cols)} sensor columns")
        return self

    def transform(self, X):
        """Transform raw sensor data into engineered features.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature DataFrame with sensor readings.

        Returns
        -------
        pandas.DataFrame
            DataFrame with engineered features.
        """
        df = X.copy()
        original_len = len(df)

        df = self._add_rolling_statistics(df)
        df = self._add_rate_of_change(df)
        if self.include_cross_correlation:
            df = self._add_cross_sensor_features(df)
        df = self._add_time_features(df)
        df = self._add_generation_load_features(df)
        df = self._add_maintenance_features(df)

        df = df.dropna()
        self._feature_names = df.columns.tolist()

        print(f"Engineered features: {len(df.columns)} columns "
              f"({original_len - len(df)} rows dropped from rolling NaN)")
        return df

    def fit_transform(self, X, y=None):
        """Fit and transform in one step.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature DataFrame.
        y : ignored
            Not used, present for API compatibility.

        Returns
        -------
        pandas.DataFrame
            Transformed DataFrame.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        """Return the feature names after transformation.

        Returns
        -------
        list of str
            Feature name strings.
        """
        return self._feature_names or []

    def _add_rolling_statistics(self, df):
        """Compute rolling mean, std, min, max for sensor columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with rolling statistic columns appended.
        """
        for window in self.rolling_windows:
            for col in self._sensor_cols:
                rolling = df[col].rolling(window=window, min_periods=1)
                df[f"{col}_roll{window}_mean"] = rolling.mean()
                df[f"{col}_roll{window}_std"] = rolling.std()
                df[f"{col}_roll{window}_min"] = rolling.min()
                df[f"{col}_roll{window}_max"] = rolling.max()
        return df

    def _add_rate_of_change(self, df):
        """Compute first derivative (rate of change) for sensor columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with rate-of-change columns appended.
        """
        for col in self._sensor_cols:
            df[f"{col}_roc"] = df[col].diff()
            df[f"{col}_roc_abs"] = df[col].diff().abs()
        return df

    def _add_cross_sensor_features(self, df):
        """Compute cross-sensor correlation and balance features.

        Captures solar-wind balance, generation-load mismatch, and
        inter-sensor variance.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with cross-sensor feature columns appended.
        """
        solar_cols = [c for c in df.columns if c.startswith("solar_") and c.endswith("_kw")]
        wind_cols = [c for c in df.columns if c.startswith("wind_") and c.endswith("_kw")]

        if solar_cols:
            df["solar_total_kw"] = df[solar_cols].sum(axis=1)
            df["solar_std_across"] = df[solar_cols].std(axis=1)
            df["solar_min_across"] = df[solar_cols].min(axis=1)
            df["solar_max_across"] = df[solar_cols].max(axis=1)

        if wind_cols:
            df["wind_total_kw"] = df[wind_cols].sum(axis=1)
            df["wind_std_across"] = df[wind_cols].std(axis=1)

        if solar_cols and wind_cols:
            df["solar_wind_ratio"] = df["solar_total_kw"] / (df["wind_total_kw"] + 1e-6)
            df["solar_wind_balance"] = df["solar_total_kw"] - df["wind_total_kw"]

        if "total_generation_mw" in df.columns and "total_load_mw" in df.columns:
            df["gen_load_mismatch"] = df["total_generation_mw"] - df["total_load_mw"]
            df["gen_load_mismatch_abs"] = df["gen_load_mismatch"].abs()

        return df

    def _add_time_features(self, df):
        """Add cyclical time encoding features (hour sin/cos).

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with time encoding columns appended.
        """
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            hour = ts.dt.hour + ts.dt.minute / 60.0
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            df["day_of_week_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
            df["day_of_week_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
            df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
            df["is_daytime"] = ((hour >= 6) & (hour <= 18)).astype(int)
        return df

    def _add_generation_load_features(self, df):
        """Compute generation-load ratio and reserve margin features.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with generation-load features appended.
        """
        if "total_generation_mw" in df.columns and "total_load_mw" in df.columns:
            df["gen_load_ratio"] = df["total_generation_mw"] / (df["total_load_mw"] + 1e-6)
            df["reserve_margin"] = (
                (df["total_generation_mw"] - df["total_load_mw"])
                / (df["total_load_mw"] + 1e-6)
            )
            df["reserve_margin_pct"] = df["reserve_margin"] * 100
        return df

    def _add_maintenance_features(self, df):
        """Add time-since-last-maintenance proxy features.

        Uses cumulative sum of anomaly-free periods as a maintenance proxy.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with maintenance features appended.
        """
        if "grid_frequency_hz" in df.columns:
            freq_deviation = (df["grid_frequency_hz"] - 60.0).abs()
            df["freq_deviation"] = freq_deviation
            df["freq_deviation_rolling"] = freq_deviation.rolling(window=30, min_periods=1).mean()

        if "grid_voltage_v" in df.columns:
            voltage_deviation = (df["grid_voltage_v"] - 230.0).abs()
            df["voltage_deviation"] = voltage_deviation
            df["voltage_deviation_rolling"] = voltage_deviation.rolling(window=30, min_periods=1).mean()

        return df

    def select_features(self, df, exclude_prefixes=None):
        """Select only numeric engineered features, excluding raw sensor columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Engineered feature DataFrame.
        exclude_prefixes : list of str or None
            Column prefixes to exclude.

        Returns
        -------
        pandas.DataFrame
            DataFrame with selected features only.
        """
        exclude_prefixes = exclude_prefixes or []
        exclude_cols = ["timestamp", "is_anomaly", "anomaly_type"]

        selected = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in exclude_cols
                     and not any(c.startswith(p) for p in exclude_prefixes)]

        print(f"Selected {len(selected)} features from {len(df.columns)} total columns")
        return df[selected]


if __name__ == "__main__":
    print("=== Feature Engineering Demo ===")
    n = 200
    demo_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
        "solar_0_kw": np.random.uniform(0, 80, n),
        "wind_0_kw": np.random.uniform(0, 500, n),
        "grid_frequency_hz": np.random.normal(60, 0.01, n),
        "grid_voltage_v": np.random.normal(230, 0.5, n),
        "total_generation_mw": np.random.uniform(1, 5, n),
        "total_load_mw": np.random.uniform(1.5, 3, n),
    })

    engineer = GridFeatureEngineer(rolling_windows=[5, 10])
    result = engineer.fit_transform(demo_df)
    print(f"Output shape: {result.shape}")
