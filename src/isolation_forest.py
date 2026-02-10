import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class IsolationForestDetector:
    """Isolation Forest anomaly detector for energy grid data.

    Wraps scikit-learn's IsolationForest with preprocessing, threshold
    tuning, and comparison metrics against other models.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in the dataset.
    n_estimators : int
        Number of trees in the forest.
    max_samples : str or int
        Number of samples to draw for each tree.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, contamination=0.05, n_estimators=200,
                 max_samples="auto", random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X):
        """Fit the Isolation Forest model.

        Parameters
        ----------
        X : array-like
            Training feature matrix (may include both normal and anomalous).

        Returns
        -------
        IsolationForestDetector
            self
        """
        print(f"Training Isolation Forest on {len(X)} samples...")
        print(f"  Estimators: {self.n_estimators} | Contamination: {self.contamination}")

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._fitted = True

        print("Isolation Forest training complete")
        return self

    def predict(self, X):
        """Predict anomaly labels (0=normal, 1=anomaly).

        Converts sklearn's convention (-1=anomaly, 1=normal) to
        binary format (1=anomaly, 0=normal).

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Array of binary predictions.
        """
        X_scaled = self.scaler.transform(X)
        raw_preds = self.model.predict(X_scaled)
        return (raw_preds == -1).astype(int)

    def compute_anomaly_scores(self, X):
        """Compute anomaly scores for each sample.

        Higher scores indicate more anomalous behavior.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Array of anomaly scores (negated decision function).
        """
        X_scaled = self.scaler.transform(X)
        scores = -self.model.decision_function(X_scaled)
        return scores

    def evaluate(self, X, y_true):
        """Evaluate the detector against ground truth labels.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y_true : array-like
            True binary anomaly labels.

        Returns
        -------
        dict
            Precision, recall, F1, and confusion matrix.
        """
        y_pred = self.predict(X)
        results = {
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, target_names=["Normal", "Anomaly"], output_dict=True
            ),
        }

        print(f"  Precision: {results['precision']}")
        print(f"  Recall: {results['recall']}")
        print(f"  F1-Score: {results['f1_score']}")
        return results

    def tune_contamination(self, X, y_true, values=None):
        """Tune the contamination parameter using grid search.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y_true : array-like
            True binary anomaly labels.
        values : list of float or None
            List of contamination values to try.

        Returns
        -------
        dict
            Best contamination value and corresponding metrics.
        """
        values = values or [0.01, 0.02, 0.05, 0.08, 0.10, 0.15]
        best_f1 = -1
        best_contamination = self.contamination
        results = []

        print("Tuning contamination parameter...")
        for c in values:
            model = IsolationForest(
                contamination=c,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                random_state=self.random_state,
                n_jobs=-1,
            )
            X_scaled = self.scaler.transform(X)
            model.fit(X_scaled)
            raw_preds = model.predict(X_scaled)
            y_pred = (raw_preds == -1).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)

            results.append({
                "contamination": c,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
            })
            print(f"  contamination={c:.2f} | F1={f1:.4f} | P={prec:.4f} | R={rec:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_contamination = c

        print(f"\nBest contamination: {best_contamination} (F1={best_f1:.4f})")
        return {
            "best_contamination": best_contamination,
            "best_f1": round(best_f1, 4),
            "all_results": results,
        }

    def compare_with_autoencoder(self, X, y_true, autoencoder_preds):
        """Compare Isolation Forest results with autoencoder predictions.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y_true : array-like
            True binary anomaly labels.
        autoencoder_preds : array-like
            Binary predictions from the autoencoder.

        Returns
        -------
        pandas.DataFrame
            DataFrame comparing metrics for both models.
        """
        if_preds = self.predict(X)

        if_metrics = {
            "Model": "Isolation Forest",
            "Precision": round(precision_score(y_true, if_preds, zero_division=0), 4),
            "Recall": round(recall_score(y_true, if_preds, zero_division=0), 4),
            "F1-Score": round(f1_score(y_true, if_preds, zero_division=0), 4),
        }
        ae_metrics = {
            "Model": "Autoencoder",
            "Precision": round(precision_score(y_true, autoencoder_preds, zero_division=0), 4),
            "Recall": round(recall_score(y_true, autoencoder_preds, zero_division=0), 4),
            "F1-Score": round(f1_score(y_true, autoencoder_preds, zero_division=0), 4),
        }

        ensemble_preds = ((if_preds + autoencoder_preds) > 0).astype(int)
        ensemble_metrics = {
            "Model": "Ensemble (OR)",
            "Precision": round(precision_score(y_true, ensemble_preds, zero_division=0), 4),
            "Recall": round(recall_score(y_true, ensemble_preds, zero_division=0), 4),
            "F1-Score": round(f1_score(y_true, ensemble_preds, zero_division=0), 4),
        }

        comparison = pd.DataFrame([if_metrics, ae_metrics, ensemble_metrics])
        print("\nModel Comparison:")
        print(comparison.to_string(index=False))
        return comparison

    def save(self, model_dir="models"):
        """Save the trained model and scaler to disk.

        Parameters
        ----------
        model_dir : str
            Directory to save model artifacts.
        """
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "isolation_forest.pkl")
        joblib.dump(self.scaler, path / "isolation_forest_scaler.pkl")
        print(f"Isolation Forest saved to {path}")

    def load(self, model_dir="models"):
        """Load a trained model from disk.

        Parameters
        ----------
        model_dir : str
            Directory containing model artifacts.

        Returns
        -------
        IsolationForestDetector
            self
        """
        path = Path(model_dir)
        self.model = joblib.load(path / "isolation_forest.pkl")
        self.scaler = joblib.load(path / "isolation_forest_scaler.pkl")
        self._fitted = True
        print(f"Isolation Forest loaded from {path}")
        return self

    def get_model_info(self):
        """Return metadata about the trained model.

        Returns
        -------
        dict
            Model parameters.
        """
        return {
            "model_type": "IsolationForest",
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "max_samples": self.max_samples,
            "fitted": self._fitted,
        }


if __name__ == "__main__":
    print("=== Isolation Forest Demo ===")
    n_features = 20
    X_normal = np.random.randn(500, n_features)
    X_anomaly = np.random.randn(30, n_features) * 3 + 2
    X_all = np.vstack([X_normal, X_anomaly])
    y_all = np.array([0] * 500 + [1] * 30)

    detector = IsolationForestDetector(contamination=0.05)
    detector.fit(X_all)
    results = detector.evaluate(X_all, y_all)
    print(detector.get_model_info())
