import logging
import threading
import queue
import time

import numpy as np
import pandas as pd

from .logging_conf import setup as setup_logging

setup_logging()
logger = logging.getLogger("anomaly_detector")


class StreamProcessor:
    """Simulated streaming processor for grid sensor data.

    Implements a Kafka-consumer-style pattern using Python threading
    and queues. Supports sliding window aggregation and both streaming
    and batch processing modes.

    Parameters
    ----------
    window_size : int
        Number of records in each sliding window.
    step_size : int
        Number of records to advance between windows.
    max_queue_size : int
        Maximum items in the internal buffer queue.
    """

    def __init__(self, window_size=5, step_size=1, max_queue_size=10000):
        self.window_size = window_size
        self.step_size = step_size
        self.max_queue_size = max_queue_size
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._results_queue = queue.Queue()
        self._running = False
        self._consumer_thread = None
        self._window_buffer = []
        self._processed_count = 0

    def start(self):
        """Start the stream consumer thread."""
        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._consume_loop, daemon=True
        )
        self._consumer_thread.start()
        logger.info("StreamProcessor started (consumer thread running)")
        print("StreamProcessor started (consumer thread running)")

    def stop(self):
        """Stop the stream consumer thread and flush remaining data."""
        self._running = False
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=5)
        logger.info("StreamProcessor stopped | Processed %d windows", self._processed_count)
        print(f"StreamProcessor stopped | Processed {self._processed_count} windows")

    def publish(self, record):
        """Publish a single sensor record to the stream.

        Parameters
        ----------
        record : dict or pandas.Series
            One timestep of sensor data.
        """
        try:
            self._queue.put(record, timeout=1)
        except queue.Full:
            logger.warning("Stream buffer full, dropping record")
            print("WARNING: Stream buffer full, dropping record")

    def publish_batch(self, df):
        """Publish a batch of records to the stream.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of sensor records.
        """
        print(f"Publishing {len(df)} records to stream...")
        for _, row in df.iterrows():
            self.publish(row.to_dict())
        print(f"Published {len(df)} records")

    def get_results(self, timeout=1):
        """Retrieve processed window results.

        Parameters
        ----------
        timeout : int
            Seconds to wait for a result.

        Returns
        -------
        dict or None
            Aggregated window features, or None if timeout.
        """
        try:
            return self._results_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_all_results(self):
        """Retrieve all available processed results.

        Returns
        -------
        list of dict
            Aggregated window feature dicts.
        """
        results = []
        while not self._results_queue.empty():
            try:
                results.append(self._results_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _consume_loop(self):
        """Internal consumer loop running in a separate thread."""
        while self._running or not self._queue.empty():
            try:
                record = self._queue.get(timeout=0.5)
                self._window_buffer.append(record)

                if len(self._window_buffer) >= self.window_size:
                    window_df = pd.DataFrame(self._window_buffer[-self.window_size:])
                    features = self._compute_window_features(window_df)
                    self._results_queue.put(features)
                    self._processed_count += 1

                    if len(self._window_buffer) > self.window_size * 2:
                        self._window_buffer = self._window_buffer[-self.window_size:]

            except queue.Empty:
                continue

    def _compute_window_features(self, window_df):
        """Compute aggregated features for a sliding window.

        Parameters
        ----------
        window_df : pandas.DataFrame
            DataFrame containing records in the current window.

        Returns
        -------
        dict
            Statistical summaries for each numeric column.
        """
        numeric_cols = window_df.select_dtypes(include=[np.number]).columns
        features = {
            "window_start": self._processed_count * self.step_size,
            "window_size": len(window_df),
        }

        for col in numeric_cols:
            series = window_df[col].dropna()
            if len(series) == 0:
                continue
            features[f"{col}_mean"] = round(series.mean(), 4)
            features[f"{col}_std"] = round(series.std(), 4) if len(series) > 1 else 0.0
            features[f"{col}_min"] = round(series.min(), 4)
            features[f"{col}_max"] = round(series.max(), 4)
            features[f"{col}_range"] = round(series.max() - series.min(), 4)

        return features

    def process_batch(self, df, window_size=None, step_size=None):
        """Process a DataFrame in batch mode with sliding windows.

        Runs synchronously without threading for training data preparation.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with sensor readings.
        window_size : int or None
            Override instance window_size.
        step_size : int or None
            Override instance step_size.

        Returns
        -------
        pandas.DataFrame
            DataFrame of aggregated window features.
        """
        ws = window_size or self.window_size
        ss = step_size or self.step_size
        results = []

        logger.debug("Batch processing %d records (window=%d, step=%d)", len(df), ws, ss)
        print(f"Batch processing {len(df)} records (window={ws}, step={ss})...")

        for start in range(0, len(df) - ws + 1, ss):
            window = df.iloc[start:start + ws]
            features = self._compute_window_features(window)
            features["window_center_idx"] = start + ws // 2

            if "is_anomaly" in window.columns:
                features["is_anomaly"] = int(window["is_anomaly"].max())
            if "anomaly_type" in window.columns:
                types = window.loc[window["is_anomaly"] == 1, "anomaly_type"]
                features["anomaly_type"] = types.mode().iloc[0] if len(types) > 0 else "normal"

            results.append(features)

        result_df = pd.DataFrame(results)
        print(f"  Generated {len(result_df)} window features")
        return result_df

    def process_multi_window(self, df, windows=None):
        """Process data with multiple window sizes for richer features.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with sensor readings.
        windows : list of int or None
            List of window sizes. Defaults to [5, 15].

        Returns
        -------
        pandas.DataFrame
            DataFrame with features from all window sizes merged.
        """
        windows = windows or [5, 15]
        all_features = []

        for ws in windows:
            print(f"  Processing window size {ws}...")
            batch_result = self.process_batch(df, window_size=ws, step_size=1)
            renamed = batch_result.add_prefix(f"w{ws}_")
            if "window_center_idx" in batch_result.columns:
                renamed["_merge_idx"] = batch_result["window_center_idx"]
            all_features.append(renamed)

        merged = all_features[0]
        for other in all_features[1:]:
            merged = merged.merge(other, on="_merge_idx", how="inner")

        if "_merge_idx" in merged.columns:
            merged = merged.drop(columns=["_merge_idx"])

        print(f"  Multi-window features: {merged.shape[1]} columns")
        return merged


if __name__ == "__main__":
    print("=== Stream Processor Demo ===")
    demo_data = pd.DataFrame({
        "solar_0_kw": np.random.uniform(0, 80, 100),
        "wind_0_kw": np.random.uniform(0, 500, 100),
        "grid_frequency_hz": np.random.normal(60, 0.01, 100),
        "is_anomaly": np.zeros(100),
        "anomaly_type": ["normal"] * 100,
    })
    processor = StreamProcessor(window_size=5, step_size=2)
    result = processor.process_batch(demo_data)
    print(f"Result shape: {result.shape}")
