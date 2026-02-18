import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


SEVERITY_LEVELS = {
    "normal": 0,
    "warning": 1,
    "critical": 2,
}


class AlertEngine:
    """Alert engine for energy grid anomaly notifications.

    Classifies anomaly scores into severity levels, deduplicates alerts
    with configurable cooldown periods, and maintains alert history
    with summary statistics.

    Parameters
    ----------
    warning_threshold : float
        Anomaly score threshold for warning alerts.
    critical_threshold : float
        Anomaly score threshold for critical alerts.
    cooldown_seconds : int
        Minimum seconds between duplicate alerts.
    max_history : int
        Maximum number of alerts to store in history.
    """

    def __init__(self, warning_threshold=0.5, critical_threshold=0.8,
                 cooldown_seconds=300, max_history=10000):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history
        self._alert_history = []
        self._last_alert_time = {}
        self._alert_counts = defaultdict(int)
        self._suppressed_count = 0

    def classify(self, anomaly_score):
        """Classify an anomaly score into a severity level.

        Parameters
        ----------
        anomaly_score : float
            Numeric anomaly score from a detection model.

        Returns
        -------
        str
            Severity level: 'normal', 'warning', or 'critical'.
        """
        if anomaly_score >= self.critical_threshold:
            return "critical"
        elif anomaly_score >= self.warning_threshold:
            return "warning"
        return "normal"

    def process(self, anomaly_score, sensor_id=None, metadata=None):
        """Process an anomaly score and generate an alert if appropriate.

        Applies severity classification and deduplication before creating
        an alert record.

        Parameters
        ----------
        anomaly_score : float
            Numeric anomaly score.
        sensor_id : str or None
            Identifier for the sensor that produced the reading.
        metadata : dict or None
            Optional additional context.

        Returns
        -------
        dict or None
            Alert dict if generated, None if suppressed or normal.
        """
        severity = self.classify(anomaly_score)

        if severity == "normal":
            return None

        alert_key = f"{sensor_id}_{severity}"
        now = datetime.now()

        if alert_key in self._last_alert_time:
            elapsed = (now - self._last_alert_time[alert_key]).total_seconds()
            if elapsed < self.cooldown_seconds:
                self._suppressed_count += 1
                return None

        alert = {
            "timestamp": now.isoformat(),
            "severity": severity,
            "anomaly_score": round(float(anomaly_score), 4),
            "sensor_id": sensor_id or "unknown",
            "metadata": metadata or {},
            "alert_id": f"ALT-{len(self._alert_history) + 1:06d}",
        }

        self._last_alert_time[alert_key] = now
        self._alert_counts[severity] += 1
        self._alert_history.append(alert)

        if len(self._alert_history) > self.max_history:
            self._alert_history = self._alert_history[-self.max_history:]

        return alert

    def process_batch(self, scores, sensor_ids=None):
        """Process a batch of anomaly scores.

        Parameters
        ----------
        scores : array-like
            Array or list of anomaly scores.
        sensor_ids : list of str or None
            Optional list of sensor identifiers.

        Returns
        -------
        list of dict
            List of generated alerts (excluding suppressed/normal).
        """
        alerts = []
        sensor_ids = sensor_ids or [None] * len(scores)

        for score, sid in zip(scores, sensor_ids):
            alert = self.process(score, sensor_id=sid)
            if alert is not None:
                alerts.append(alert)

        print(f"Processed {len(scores)} scores | "
              f"Generated {len(alerts)} alerts | "
              f"Suppressed {self._suppressed_count}")
        return alerts

    def get_recent_alerts(self, n=20):
        """Retrieve the most recent alerts.

        Parameters
        ----------
        n : int
            Number of recent alerts to return.

        Returns
        -------
        list of dict
            Recent alert dicts.
        """
        return self._alert_history[-n:]

    def get_alerts_by_severity(self, severity):
        """Retrieve alerts filtered by severity level.

        Parameters
        ----------
        severity : str
            Severity level string ('warning' or 'critical').

        Returns
        -------
        list of dict
            Matching alert dicts.
        """
        return [a for a in self._alert_history if a["severity"] == severity]

    def get_summary(self):
        """Generate summary statistics for all alerts.

        Returns
        -------
        dict
            Alert counts, rates, and breakdown by severity.
        """
        total = len(self._alert_history)
        if total == 0:
            return {
                "total_alerts": 0,
                "suppressed": self._suppressed_count,
                "by_severity": {},
                "alerts_per_hour": 0,
            }

        timestamps = [datetime.fromisoformat(a["timestamp"]) for a in self._alert_history]
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
        alerts_per_hour = total / max(time_span, 1 / 3600)

        severity_breakdown = {}
        for severity in ["warning", "critical"]:
            count = self._alert_counts.get(severity, 0)
            severity_breakdown[severity] = {
                "count": count,
                "percentage": round(count / total * 100, 1) if total > 0 else 0,
            }

        sensor_counts = defaultdict(int)
        for alert in self._alert_history:
            sensor_counts[alert["sensor_id"]] += 1
        top_sensors = sorted(sensor_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        summary = {
            "total_alerts": total,
            "suppressed": self._suppressed_count,
            "alerts_per_hour": round(alerts_per_hour, 2),
            "by_severity": severity_breakdown,
            "top_sensors": [{"sensor_id": s, "count": c} for s, c in top_sensors],
        }
        return summary

    def print_summary(self):
        """Print a formatted summary of alert statistics."""
        summary = self.get_summary()
        print("\n=== Alert Engine Summary ===")
        print(f"Total alerts:     {summary['total_alerts']}")
        print(f"Suppressed:       {summary['suppressed']}")
        print(f"Alerts per hour:  {summary['alerts_per_hour']}")
        print(f"\nBy Severity:")
        for severity, info in summary.get("by_severity", {}).items():
            print(f"  {severity.upper()}: {info['count']} ({info['percentage']}%)")
        if summary.get("top_sensors"):
            print(f"\nTop Sensors:")
            for sensor in summary["top_sensors"]:
                print(f"  {sensor['sensor_id']}: {sensor['count']} alerts")

    def reset(self):
        """Clear all alert history and counters."""
        self._alert_history = []
        self._last_alert_time = {}
        self._alert_counts = defaultdict(int)
        self._suppressed_count = 0
        print("Alert engine reset")

    def update_thresholds(self, warning=None, critical=None):
        """Update alert severity thresholds.

        Parameters
        ----------
        warning : float or None
            New warning threshold (or None to keep current).
        critical : float or None
            New critical threshold (or None to keep current).
        """
        if warning is not None:
            self.warning_threshold = warning
        if critical is not None:
            self.critical_threshold = critical
        print(f"Thresholds updated | Warning: {self.warning_threshold} | "
              f"Critical: {self.critical_threshold}")

    def get_alert_rate_history(self, window_minutes=60):
        """Compute alert rate over time windows.

        Parameters
        ----------
        window_minutes : int
            Size of each time window in minutes.

        Returns
        -------
        list of dict
            Dicts with window timestamps and alert counts.
        """
        if not self._alert_history:
            return []

        timestamps = [datetime.fromisoformat(a["timestamp"]) for a in self._alert_history]
        min_time = min(timestamps)
        max_time = max(timestamps)
        window = timedelta(minutes=window_minutes)

        buckets = []
        current = min_time
        while current <= max_time:
            bucket_end = current + window
            count = sum(1 for ts in timestamps if current <= ts < bucket_end)
            buckets.append({
                "window_start": current.isoformat(),
                "window_end": bucket_end.isoformat(),
                "alert_count": count,
            })
            current = bucket_end

        return buckets


if __name__ == "__main__":
    print("=== Alert Engine Demo ===")
    engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8, cooldown_seconds=0)

    scores = np.concatenate([
        np.random.uniform(0, 0.3, 50),   # normal
        np.random.uniform(0.5, 0.7, 20),  # warning
        np.random.uniform(0.8, 1.0, 10),  # critical
    ])
    sensor_ids = [f"sensor_{i % 10}" for i in range(len(scores))]

    alerts = engine.process_batch(scores, sensor_ids)
    engine.print_summary()
