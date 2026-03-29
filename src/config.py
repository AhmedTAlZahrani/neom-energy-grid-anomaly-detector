"""Detector configuration with dataclass validation."""

from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Configuration for the energy grid anomaly detector.

    Parameters
    ----------
    window_size : int
        Number of timesteps in the sliding detection window.
    anomaly_threshold : float
        Reconstruction error threshold for flagging anomalies.
    model_path : str
        Path to the trained autoencoder weights.
    alert_cooldown_seconds : int
        Minimum seconds between consecutive alerts for the same zone.
    warning_threshold : float
        Deviation level that triggers a warning.
    critical_threshold : float
        Deviation level that triggers a critical alert.
    """

    window_size: int = 60
    anomaly_threshold: float = 0.05
    model_path: str = "models/autoencoder.keras"
    alert_cooldown_seconds: int = 300
    warning_threshold: float = 0.03
    critical_threshold: float = 0.08

    def __post_init__(self):
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.warning_threshold >= self.critical_threshold:
            raise ValueError(
                f"warning_threshold ({self.warning_threshold}) must be less than "
                f"critical_threshold ({self.critical_threshold})"
            )
        if self.anomaly_threshold < 0:
            raise ValueError(f"anomaly_threshold must be non-negative, got {self.anomaly_threshold}")
        if self.alert_cooldown_seconds < 0:
            raise ValueError(f"alert_cooldown_seconds must be non-negative, got {self.alert_cooldown_seconds}")
