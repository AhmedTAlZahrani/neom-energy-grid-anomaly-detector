"""Tests for src.stream_processor.StreamProcessor.

Covers batch processing, sliding-window feature computation, multi-window
merge, and the threaded start/stop/publish lifecycle.
"""
import time

import numpy as np
import pandas as pd
import pytest

from src.stream_processor import StreamProcessor


# ---------------------------------------------------------------------------
# Batch processing (synchronous, no threading)
# ---------------------------------------------------------------------------

class TestProcessBatch:
    """Batch-mode sliding window tests."""

    def test_returns_dataframe(self, sample_sensor_df):
        proc = StreamProcessor(window_size=5, step_size=2)
        result = proc.process_batch(sample_sensor_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_window_count(self, sample_sensor_df):
        ws, ss = 5, 2
        proc = StreamProcessor(window_size=ws, step_size=ss)
        result = proc.process_batch(sample_sensor_df)
        expected = len(range(0, len(sample_sensor_df) - ws + 1, ss))
        assert len(result) == expected

    def test_window_size_column(self, sample_sensor_df):
        proc = StreamProcessor(window_size=10, step_size=5)
        result = proc.process_batch(sample_sensor_df)
        assert (result["window_size"] == 10).all()

    def test_feature_columns_present(self, sample_sensor_df):
        proc = StreamProcessor(window_size=5, step_size=1)
        result = proc.process_batch(sample_sensor_df)
        # numeric columns should produce _mean, _std, _min, _max, _range
        for suffix in ("_mean", "_std", "_min", "_max", "_range"):
            matching = [c for c in result.columns if c.endswith(suffix)]
            assert len(matching) > 0, f"No columns ending with {suffix}"

    def test_anomaly_label_propagation(self, sample_sensor_df):
        proc = StreamProcessor(window_size=5, step_size=1)
        result = proc.process_batch(sample_sensor_df)
        assert "is_anomaly" in result.columns
        assert "anomaly_type" in result.columns
        # at least some windows should be anomalous
        assert result["is_anomaly"].sum() > 0

    def test_override_window_params(self, sample_sensor_df):
        proc = StreamProcessor(window_size=5, step_size=1)
        result = proc.process_batch(sample_sensor_df, window_size=10, step_size=3)
        expected = len(range(0, len(sample_sensor_df) - 10 + 1, 3))
        assert len(result) == expected

    def test_single_numeric_col(self):
        df = pd.DataFrame({"value": np.arange(20, dtype=float)})
        proc = StreamProcessor(window_size=5, step_size=5)
        result = proc.process_batch(df)
        assert "value_mean" in result.columns
        assert "value_range" in result.columns

    def test_empty_after_nan_columns_skipped(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "label": ["a", "b", "c", "d", "e"],
        })
        proc = StreamProcessor(window_size=3, step_size=1)
        result = proc.process_batch(df)
        # string column should not produce features
        str_features = [c for c in result.columns if c.startswith("label_")]
        assert len(str_features) == 0


# ---------------------------------------------------------------------------
# Window feature math
# ---------------------------------------------------------------------------

class TestComputeWindowFeatures:
    """Unit tests for _compute_window_features."""

    def test_mean_value(self):
        proc = StreamProcessor()
        window = pd.DataFrame({"v": [2.0, 4.0, 6.0]})
        features = proc._compute_window_features(window)
        assert features["v_mean"] == pytest.approx(4.0, abs=1e-3)

    def test_range_value(self):
        proc = StreamProcessor()
        window = pd.DataFrame({"v": [1.0, 5.0, 3.0]})
        features = proc._compute_window_features(window)
        assert features["v_range"] == pytest.approx(4.0, abs=1e-3)

    def test_std_single_row(self):
        proc = StreamProcessor()
        window = pd.DataFrame({"v": [7.0]})
        features = proc._compute_window_features(window)
        assert features["v_std"] == 0.0

    def test_all_nan_column_skipped(self):
        proc = StreamProcessor()
        window = pd.DataFrame({"v": [np.nan, np.nan, np.nan]})
        features = proc._compute_window_features(window)
        assert "v_mean" not in features


# ---------------------------------------------------------------------------
# Multi-window processing
# ---------------------------------------------------------------------------

class TestProcessMultiWindow:

    def test_returns_dataframe(self, sample_sensor_df):
        proc = StreamProcessor(window_size=5, step_size=1)
        result = proc.process_multi_window(sample_sensor_df, windows=[5, 10])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_columns_prefixed(self, sample_sensor_df):
        proc = StreamProcessor(window_size=5, step_size=1)
        result = proc.process_multi_window(sample_sensor_df, windows=[5, 10])
        w5_cols = [c for c in result.columns if c.startswith("w5_")]
        w10_cols = [c for c in result.columns if c.startswith("w10_")]
        assert len(w5_cols) > 0
        assert len(w10_cols) > 0

    def test_merge_idx_dropped(self, sample_sensor_df):
        proc = StreamProcessor(window_size=5, step_size=1)
        result = proc.process_multi_window(sample_sensor_df, windows=[5, 10])
        assert "_merge_idx" not in result.columns


# ---------------------------------------------------------------------------
# Threaded streaming (start / stop / publish)
# ---------------------------------------------------------------------------

class TestStreamingLifecycle:

    @pytest.mark.timeout(10)
    def test_start_stop(self):
        proc = StreamProcessor(window_size=3, step_size=1)
        proc.start()
        assert proc._running is True
        assert proc._consumer_thread is not None
        assert proc._consumer_thread.is_alive()
        proc.stop()
        assert proc._running is False

    @pytest.mark.timeout(10)
    def test_publish_and_consume(self, stream_records):
        proc = StreamProcessor(window_size=3, step_size=1, max_queue_size=100)
        proc.start()

        for rec in stream_records:
            proc.publish(rec)

        # give consumer thread time to drain the queue
        time.sleep(1.5)
        proc.stop()

        results = proc.get_all_results()
        assert len(results) > 0
        # each result should have window_size key
        assert all("window_size" in r for r in results)

    @pytest.mark.timeout(10)
    def test_get_results_returns_none_on_empty(self):
        proc = StreamProcessor(window_size=3)
        proc.start()
        result = proc.get_results(timeout=0.3)
        assert result is None
        proc.stop()

    @pytest.mark.timeout(10)
    def test_processed_count_increments(self, stream_records):
        proc = StreamProcessor(window_size=3, step_size=1)
        proc.start()

        for rec in stream_records:
            proc.publish(rec)

        time.sleep(1.5)
        proc.stop()
        assert proc._processed_count > 0

    @pytest.mark.timeout(10)
    def test_publish_batch_method(self, sample_sensor_df):
        small_df = sample_sensor_df.head(15)
        proc = StreamProcessor(window_size=3, step_size=1)
        proc.start()
        proc.publish_batch(small_df)
        time.sleep(1.5)
        proc.stop()

        results = proc.get_all_results()
        assert len(results) > 0

    @pytest.mark.timeout(10)
    def test_queue_full_does_not_raise(self):
        proc = StreamProcessor(window_size=3, max_queue_size=2)
        # do NOT start consumer -- queue will fill up
        for _ in range(5):
            proc.publish({"v": 1.0})
        # should not raise; just prints warning and drops
