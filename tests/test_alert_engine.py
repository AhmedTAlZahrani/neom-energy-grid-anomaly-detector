"""Tests for src.alert_engine.AlertEngine.

Covers severity classification, cooldown deduplication, batch processing,
summary statistics, threshold updates, and alert-rate bucketing.
"""
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest

from src.alert_engine import AlertEngine, SEVERITY_LEVELS


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

class TestClassify:

    def test_normal_below_warning(self):
        engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8)
        assert engine.classify(0.0) == "normal"
        assert engine.classify(0.49) == "normal"

    def test_warning_range(self):
        engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8)
        assert engine.classify(0.5) == "warning"
        assert engine.classify(0.6) == "warning"
        assert engine.classify(0.79) == "warning"

    def test_critical_at_threshold(self):
        engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8)
        assert engine.classify(0.8) == "critical"
        assert engine.classify(1.0) == "critical"

    def test_negative_score_is_normal(self):
        engine = AlertEngine(warning_threshold=0.5)
        assert engine.classify(-0.1) == "normal"

    def test_custom_thresholds(self):
        engine = AlertEngine(warning_threshold=0.3, critical_threshold=0.6)
        assert engine.classify(0.35) == "warning"
        assert engine.classify(0.65) == "critical"

    def test_boundary_values(self, anomaly_scores_mixed):
        engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8)
        labels = [engine.classify(s) for s in anomaly_scores_mixed]
        assert labels.count("normal") == 5
        assert labels.count("warning") == 5
        assert labels.count("critical") == 5


# ---------------------------------------------------------------------------
# Process (single alert generation)
# ---------------------------------------------------------------------------

class TestProcess:

    def test_normal_score_returns_none(self):
        engine = AlertEngine(warning_threshold=0.5)
        result = engine.process(0.2, sensor_id="s1")
        assert result is None

    def test_warning_generates_alert(self):
        engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8,
                             cooldown_seconds=0)
        alert = engine.process(0.6, sensor_id="s1")
        assert alert is not None
        assert alert["severity"] == "warning"
        assert alert["sensor_id"] == "s1"
        assert "timestamp" in alert
        assert "alert_id" in alert

    def test_critical_generates_alert(self):
        engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8,
                             cooldown_seconds=0)
        alert = engine.process(0.9, sensor_id="s2")
        assert alert is not None
        assert alert["severity"] == "critical"

    def test_anomaly_score_rounded(self):
        engine = AlertEngine(cooldown_seconds=0)
        alert = engine.process(0.87654321, sensor_id="x")
        assert alert["anomaly_score"] == 0.8765

    def test_default_sensor_id(self):
        engine = AlertEngine(cooldown_seconds=0)
        alert = engine.process(0.9)
        assert alert["sensor_id"] == "unknown"

    def test_metadata_attached(self):
        engine = AlertEngine(cooldown_seconds=0)
        meta = {"zone": "A1", "subsystem": "solar"}
        alert = engine.process(0.9, sensor_id="s1", metadata=meta)
        assert alert["metadata"] == meta

    def test_alert_id_increments(self):
        engine = AlertEngine(cooldown_seconds=0)
        a1 = engine.process(0.9, sensor_id="s1")
        a2 = engine.process(0.9, sensor_id="s2")
        assert a1["alert_id"] == "ALT-000001"
        assert a2["alert_id"] == "ALT-000002"


# ---------------------------------------------------------------------------
# Cooldown / deduplication
# ---------------------------------------------------------------------------

class TestCooldownDeduplication:

    def test_duplicate_suppressed_within_cooldown(self):
        engine = AlertEngine(cooldown_seconds=300)
        first = engine.process(0.9, sensor_id="s1")
        second = engine.process(0.9, sensor_id="s1")
        assert first is not None
        assert second is None

    def test_different_sensors_not_suppressed(self):
        engine = AlertEngine(cooldown_seconds=300)
        a1 = engine.process(0.9, sensor_id="s1")
        a2 = engine.process(0.9, sensor_id="s2")
        assert a1 is not None
        assert a2 is not None

    def test_different_severity_same_sensor_not_suppressed(self):
        engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8,
                             cooldown_seconds=300)
        a1 = engine.process(0.6, sensor_id="s1")  # warning
        a2 = engine.process(0.9, sensor_id="s1")  # critical
        assert a1 is not None
        assert a2 is not None

    def test_suppressed_count_tracks(self):
        engine = AlertEngine(cooldown_seconds=300)
        engine.process(0.9, sensor_id="s1")
        engine.process(0.9, sensor_id="s1")
        engine.process(0.9, sensor_id="s1")
        assert engine._suppressed_count == 2

    def test_alert_allowed_after_cooldown_expires(self):
        engine = AlertEngine(cooldown_seconds=1)
        a1 = engine.process(0.9, sensor_id="s1")
        assert a1 is not None
        # manually backdate the last alert time to simulate cooldown expiry
        key = "s1_critical"
        engine._last_alert_time[key] = datetime.now() - timedelta(seconds=5)
        a2 = engine.process(0.9, sensor_id="s1")
        assert a2 is not None


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class TestProcessBatch:

    def test_returns_list_of_alerts(self):
        engine = AlertEngine(cooldown_seconds=0)
        scores = [0.1, 0.6, 0.9]
        alerts = engine.process_batch(scores, sensor_ids=["a", "b", "c"])
        assert isinstance(alerts, list)
        # 0.1 is normal -> no alert; 0.6 and 0.9 generate alerts
        assert len(alerts) == 2

    def test_batch_without_sensor_ids(self):
        engine = AlertEngine(cooldown_seconds=0)
        scores = [0.6, 0.9]
        alerts = engine.process_batch(scores)
        assert len(alerts) == 2
        assert all(a["sensor_id"] == "unknown" for a in alerts)

    def test_batch_all_normal(self):
        engine = AlertEngine(warning_threshold=0.5)
        scores = [0.0, 0.1, 0.2, 0.3]
        alerts = engine.process_batch(scores)
        assert len(alerts) == 0

    def test_batch_with_cooldown_suppression(self):
        engine = AlertEngine(cooldown_seconds=300)
        scores = [0.9, 0.9, 0.9]
        sensor_ids = ["s1", "s1", "s1"]
        alerts = engine.process_batch(scores, sensor_ids=sensor_ids)
        # first passes, second and third suppressed
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# History and retrieval
# ---------------------------------------------------------------------------

class TestHistoryRetrieval:

    def _populated_engine(self):
        engine = AlertEngine(cooldown_seconds=0)
        sensors = [f"sensor_{i}" for i in range(5)]
        for s in sensors:
            engine.process(0.6, sensor_id=s)
            engine.process(0.9, sensor_id=s)
        return engine

    def test_get_recent_alerts(self):
        engine = self._populated_engine()
        recent = engine.get_recent_alerts(n=3)
        assert len(recent) == 3

    def test_get_alerts_by_severity(self):
        engine = self._populated_engine()
        warnings = engine.get_alerts_by_severity("warning")
        criticals = engine.get_alerts_by_severity("critical")
        assert all(a["severity"] == "warning" for a in warnings)
        assert all(a["severity"] == "critical" for a in criticals)
        assert len(warnings) == 5
        assert len(criticals) == 5

    def test_max_history_truncation(self):
        engine = AlertEngine(cooldown_seconds=0, max_history=5)
        for i in range(20):
            engine.process(0.9, sensor_id=f"sensor_{i}")
        assert len(engine._alert_history) <= 5


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

class TestGetSummary:

    def test_empty_summary(self):
        engine = AlertEngine()
        summary = engine.get_summary()
        assert summary["total_alerts"] == 0
        assert summary["alerts_per_hour"] == 0

    def test_populated_summary(self):
        engine = AlertEngine(cooldown_seconds=0)
        engine.process(0.6, sensor_id="s1")
        engine.process(0.9, sensor_id="s2")
        summary = engine.get_summary()
        assert summary["total_alerts"] == 2
        assert "warning" in summary["by_severity"]
        assert "critical" in summary["by_severity"]
        assert summary["by_severity"]["warning"]["count"] == 1
        assert summary["by_severity"]["critical"]["count"] == 1

    def test_top_sensors(self):
        engine = AlertEngine(cooldown_seconds=0)
        for _ in range(5):
            engine.process(0.9, sensor_id="hot_sensor")
        for _ in range(2):
            engine.process(0.9, sensor_id="cold_sensor")
        summary = engine.get_summary()
        top = summary["top_sensors"]
        assert top[0]["sensor_id"] == "hot_sensor"
        assert top[0]["count"] == 5


# ---------------------------------------------------------------------------
# Threshold updates and reset
# ---------------------------------------------------------------------------

class TestUpdateAndReset:

    def test_update_warning_only(self):
        engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8)
        engine.update_thresholds(warning=0.3)
        assert engine.warning_threshold == 0.3
        assert engine.critical_threshold == 0.8

    def test_update_critical_only(self):
        engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8)
        engine.update_thresholds(critical=0.7)
        assert engine.critical_threshold == 0.7
        assert engine.warning_threshold == 0.5

    def test_update_both(self):
        engine = AlertEngine()
        engine.update_thresholds(warning=0.2, critical=0.6)
        assert engine.warning_threshold == 0.2
        assert engine.critical_threshold == 0.6

    def test_reset_clears_state(self):
        engine = AlertEngine(cooldown_seconds=0)
        engine.process(0.9, sensor_id="s1")
        engine.process(0.6, sensor_id="s2")
        engine.reset()
        assert len(engine._alert_history) == 0
        assert engine._suppressed_count == 0
        assert len(engine._last_alert_time) == 0
        assert sum(engine._alert_counts.values()) == 0


# ---------------------------------------------------------------------------
# Alert rate history bucketing
# ---------------------------------------------------------------------------

class TestAlertRateHistory:

    def test_empty_returns_empty(self):
        engine = AlertEngine()
        assert engine.get_alert_rate_history() == []

    def test_buckets_contain_counts(self):
        engine = AlertEngine(cooldown_seconds=0)
        for i in range(10):
            engine.process(0.9, sensor_id=f"s{i}")
        buckets = engine.get_alert_rate_history(window_minutes=60)
        assert len(buckets) >= 1
        total_in_buckets = sum(b["alert_count"] for b in buckets)
        assert total_in_buckets == 10

    def test_bucket_keys(self):
        engine = AlertEngine(cooldown_seconds=0)
        engine.process(0.9, sensor_id="s1")
        buckets = engine.get_alert_rate_history()
        bucket = buckets[0]
        assert "window_start" in bucket
        assert "window_end" in bucket
        assert "alert_count" in bucket


# ---------------------------------------------------------------------------
# SEVERITY_LEVELS constant
# ---------------------------------------------------------------------------

class TestSeverityLevelsConstant:

    def test_all_levels_present(self):
        assert "normal" in SEVERITY_LEVELS
        assert "warning" in SEVERITY_LEVELS
        assert "critical" in SEVERITY_LEVELS

    def test_ordering(self):
        assert SEVERITY_LEVELS["normal"] < SEVERITY_LEVELS["warning"]
        assert SEVERITY_LEVELS["warning"] < SEVERITY_LEVELS["critical"]
