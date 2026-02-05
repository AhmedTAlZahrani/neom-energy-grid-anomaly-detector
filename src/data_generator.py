import numpy as np
import pandas as pd
from pathlib import Path


ANOMALY_TYPES = [
    "sensor_drift",
    "sudden_failure",
    "gradual_degradation",
    "cyber_attack",
    "dust_storm",
    "calibration_error",
    "overload",
    "communication_dropout",
]


class NEOMGridDataGenerator:
    """Generate synthetic sensor data for a NEOM renewable energy microgrid.

    Simulates 30 days of 1-minute interval readings from solar arrays,
    wind turbines, battery banks, hydrogen electrolyzers, and grid-level
    sensors. Injects configurable anomaly events across 8 categories.

    Parameters
    ----------
    n_solar : int
        Number of solar array sensors.
    n_wind : int
        Number of wind turbine sensors.
    n_battery : int
        Number of battery bank sensors.
    n_hydrogen : int
        Number of hydrogen electrolyzer sensors.
    days : int
        Number of days to simulate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_solar=50, n_wind=20, n_battery=10,
                 n_hydrogen=5, days=30, seed=42):
        self.n_solar = n_solar
        self.n_wind = n_wind
        self.n_battery = n_battery
        self.n_hydrogen = n_hydrogen
        self.days = days
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate(self, output_dir="data"):
        """Generate the full microgrid dataset and save to CSV.

        Parameters
        ----------
        output_dir : str
            Directory to save the output CSV files.

        Returns
        -------
        pandas.DataFrame
            DataFrame with all sensor readings and anomaly labels.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamps = pd.date_range(
            start="2026-01-01", periods=self.days * 24 * 60, freq="min"
        )
        n_steps = len(timestamps)
        print(f"Generating {n_steps} timesteps ({self.days} days at 1-min intervals)...")

        hours = timestamps.hour + timestamps.minute / 60.0
        day_of_year = timestamps.dayofyear

        df = pd.DataFrame({"timestamp": timestamps})

        # --- Solar arrays ---
        print(f"  Generating {self.n_solar} solar arrays...")
        for i in range(self.n_solar):
            solar_angle = np.clip(np.cos((hours - 12) * np.pi / 7), 0, 1)
            cloud_factor = 1.0 - 0.3 * self.rng.random(n_steps)
            dust_factor = 1.0 - 0.05 * self.rng.random(n_steps)
            temp_ambient = 25 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            temp_derating = np.clip(1.0 - 0.004 * (temp_ambient - 25), 0.8, 1.0)
            base_capacity = 80 + self.rng.uniform(-5, 5)
            output_kw = base_capacity * solar_angle * cloud_factor * dust_factor * temp_derating
            output_kw += self.rng.normal(0, 0.5, n_steps)
            output_kw = np.clip(output_kw, 0, None)
            df[f"solar_{i}_kw"] = np.round(output_kw, 2)

        # --- Wind turbines ---
        print(f"  Generating {self.n_wind} wind turbines...")
        for i in range(self.n_wind):
            seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
            wind_speed = self.rng.weibull(2.0, n_steps) * 6.0 * seasonal
            diurnal = 1.0 + 0.15 * np.sin(2 * np.pi * (hours - 14) / 24)
            wind_speed = wind_speed * diurnal
            cut_in, rated, cut_out = 3.0, 12.0, 25.0
            output_kw = np.where(
                wind_speed < cut_in, 0,
                np.where(
                    wind_speed < rated,
                    500 * ((wind_speed - cut_in) / (rated - cut_in)) ** 3,
                    np.where(wind_speed < cut_out, 500, 0)
                )
            )
            output_kw += self.rng.normal(0, 2.0, n_steps)
            output_kw = np.clip(output_kw, 0, 500)
            df[f"wind_{i}_kw"] = np.round(output_kw, 2)

        # --- Battery banks ---
        print(f"  Generating {self.n_battery} battery banks...")
        for i in range(self.n_battery):
            soc = np.zeros(n_steps)
            soc[0] = self.rng.uniform(40, 80)
            charge_rate = np.zeros(n_steps)
            for t in range(1, n_steps):
                solar_hour = (hours[t] >= 6) and (hours[t] <= 18)
                if solar_hour:
                    rate = self.rng.uniform(0.5, 2.0)
                else:
                    rate = self.rng.uniform(-2.0, -0.3)
                rate += self.rng.normal(0, 0.1)
                new_soc = soc[t - 1] + rate
                soc[t] = np.clip(new_soc, 20, 95)
                charge_rate[t] = soc[t] - soc[t - 1]
            df[f"battery_{i}_soc"] = np.round(soc, 2)
            df[f"battery_{i}_charge_rate"] = np.round(charge_rate, 3)

        # --- Hydrogen electrolyzers ---
        print(f"  Generating {self.n_hydrogen} hydrogen electrolyzers...")
        for i in range(self.n_hydrogen):
            excess_solar = np.clip(np.cos((hours - 12) * np.pi / 7), 0, 1)
            production_rate = excess_solar * self.rng.uniform(8, 12) + self.rng.normal(0, 0.3, n_steps)
            production_rate = np.clip(production_rate, 0, 15)
            efficiency = 0.65 + 0.1 * excess_solar + self.rng.normal(0, 0.01, n_steps)
            efficiency = np.clip(efficiency, 0.5, 0.8)
            temperature = 60 + 20 * excess_solar + self.rng.normal(0, 1.5, n_steps)
            df[f"hydrogen_{i}_prod_rate"] = np.round(production_rate, 3)
            df[f"hydrogen_{i}_efficiency"] = np.round(efficiency, 4)
            df[f"hydrogen_{i}_temp"] = np.round(temperature, 1)

        # --- Grid-level sensors ---
        print("  Generating grid-level sensors...")
        total_solar = df[[c for c in df.columns if c.startswith("solar_")]].sum(axis=1) / 1000
        total_wind = df[[c for c in df.columns if c.startswith("wind_")]].sum(axis=1) / 1000
        total_gen = total_solar + total_wind
        base_load = 2.0 + 0.8 * np.sin(2 * np.pi * (hours - 14) / 24)
        load_noise = self.rng.normal(0, 0.05, n_steps)
        total_load = base_load + load_noise
        gen_load_diff = total_gen - total_load
        frequency = 60.0 + 0.01 * gen_load_diff + self.rng.normal(0, 0.005, n_steps)
        voltage = 230.0 + 2.0 * gen_load_diff + self.rng.normal(0, 0.5, n_steps)

        df["grid_frequency_hz"] = np.round(frequency, 4)
        df["grid_voltage_v"] = np.round(voltage, 2)
        df["total_load_mw"] = np.round(total_load, 4)
        df["total_generation_mw"] = np.round(total_gen, 4)

        # --- Labels ---
        df["is_anomaly"] = 0
        df["anomaly_type"] = "normal"

        # --- Inject anomalies ---
        df = self._inject_anomalies(df, n_steps, hours)

        # --- Save ---
        csv_path = output_path / "neom_grid_data.csv"
        df.to_csv(csv_path, index=False)
        anomaly_count = df["is_anomaly"].sum()
        print(f"Dataset saved to {csv_path}")
        print(f"  Total records: {len(df)} | Anomalies: {anomaly_count} "
              f"({anomaly_count / len(df):.2%})")
        return df

    def _inject_anomalies(self, df, n_steps, hours):
        """Inject 200 anomaly events of 8 types into the dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            The sensor DataFrame to modify.
        n_steps : int
            Total number of timesteps.
        hours : array-like
            Hour-of-day array for each timestep.

        Returns
        -------
        pandas.DataFrame
            Modified DataFrame with anomalies injected.
        """
        n_anomalies = 200
        events_per_type = n_anomalies // len(ANOMALY_TYPES)
        solar_cols = [c for c in df.columns if c.startswith("solar_") and c.endswith("_kw")]
        wind_cols = [c for c in df.columns if c.startswith("wind_") and c.endswith("_kw")]
        all_sensor_cols = solar_cols + wind_cols

        print(f"  Injecting {n_anomalies} anomaly events across {len(ANOMALY_TYPES)} types...")

        for anomaly_type in ANOMALY_TYPES:
            for _ in range(events_per_type):
                start = self.rng.randint(100, n_steps - 120)
                duration = self.rng.randint(10, 60)
                end = min(start + duration, n_steps)
                idx = range(start, end)

                if anomaly_type == "sensor_drift":
                    col = self.rng.choice(all_sensor_cols)
                    drift = np.linspace(0, self.rng.uniform(10, 40), len(idx))
                    df.loc[start:end - 1, col] = df.loc[start:end - 1, col] + drift

                elif anomaly_type == "sudden_failure":
                    col = self.rng.choice(all_sensor_cols)
                    df.loc[start:end - 1, col] = 0.0

                elif anomaly_type == "gradual_degradation":
                    col = self.rng.choice(all_sensor_cols)
                    factor = np.linspace(1.0, self.rng.uniform(0.3, 0.6), len(idx))
                    df.loc[start:end - 1, col] = df.loc[start:end - 1, col] * factor

                elif anomaly_type == "cyber_attack":
                    n_affected = self.rng.randint(3, 8)
                    affected = self.rng.choice(all_sensor_cols, size=min(n_affected, len(all_sensor_cols)), replace=False)
                    for col in affected:
                        manipulation = self.rng.uniform(0.5, 1.5) * df.loc[start:end - 1, col].mean()
                        df.loc[start:end - 1, col] = manipulation

                elif anomaly_type == "dust_storm":
                    for col in solar_cols:
                        reduction = self.rng.uniform(0.05, 0.2)
                        df.loc[start:end - 1, col] = df.loc[start:end - 1, col] * reduction

                elif anomaly_type == "calibration_error":
                    col = self.rng.choice(all_sensor_cols)
                    offset = self.rng.uniform(5, 25)
                    noise = self.rng.normal(0, 3, len(idx))
                    df.loc[start:end - 1, col] = df.loc[start:end - 1, col] + offset + noise

                elif anomaly_type == "overload":
                    deviation = self.rng.uniform(0.3, 1.5)
                    direction = self.rng.choice([-1, 1])
                    df.loc[start:end - 1, "grid_frequency_hz"] += direction * deviation

                elif anomaly_type == "communication_dropout":
                    n_cols = self.rng.randint(2, 6)
                    affected = self.rng.choice(all_sensor_cols, size=min(n_cols, len(all_sensor_cols)), replace=False)
                    for col in affected:
                        df.loc[start:end - 1, col] = np.nan

                df.loc[start:end - 1, "is_anomaly"] = 1
                df.loc[start:end - 1, "anomaly_type"] = anomaly_type

        return df

    def get_anomaly_summary(self, df):
        """Print a summary of anomaly events in the dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with anomaly labels.

        Returns
        -------
        pandas.DataFrame
            DataFrame with anomaly counts per type.
        """
        anomalies = df[df["is_anomaly"] == 1]
        summary = anomalies.groupby("anomaly_type").size().reset_index(name="count")
        summary = summary.sort_values("count", ascending=False)
        print("\nAnomaly Summary:")
        for _, row in summary.iterrows():
            print(f"  {row['anomaly_type']}: {row['count']} records")
        return summary


if __name__ == "__main__":
    generator = NEOMGridDataGenerator()
    data = generator.generate()
    generator.get_anomaly_summary(data)
