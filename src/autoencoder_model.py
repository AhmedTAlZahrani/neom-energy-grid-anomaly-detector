import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class Autoencoder(nn.Module):
    """PyTorch autoencoder network for reconstruction-based anomaly detection.

    Architecture: input -> 64 -> 32 -> 16 (bottleneck) -> 32 -> 64 -> input

    Parameters
    ----------
    input_dim : int
        Number of input features.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        """Forward pass through encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of same shape.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class GridAutoencoder:
    """Autoencoder-based anomaly detector for energy grid data.

    Trains on normal operating data only. Uses reconstruction error
    as the anomaly score -- high error indicates anomalous behavior.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    learning_rate : float
        Optimizer learning rate.
    batch_size : int
        Training batch size.
    epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    threshold_percentile : int
        Percentile of validation errors for threshold.
    """

    def __init__(self, input_dim=50, learning_rate=0.001, batch_size=256,
                 epochs=100, patience=10, threshold_percentile=95):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.threshold_percentile = threshold_percentile
        self.model = Autoencoder(input_dim)
        self.scaler = StandardScaler()
        self.threshold = None
        self.training_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X_train, X_val=None):
        """Train the autoencoder on normal operating data.

        Parameters
        ----------
        X_train : array-like
            Training features (normal data only).
        X_val : array-like or None
            Validation features for early stopping and threshold tuning.

        Returns
        -------
        GridAutoencoder
            self
        """
        print(f"Training autoencoder on {len(X_train)} samples...")
        print(f"  Device: {self.device} | Input dim: {self.input_dim}")

        X_train_scaled = self.scaler.fit_transform(X_train)
        train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for batch_x, batch_target in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # print(f"Reconstruction error: {loss:.4f}")
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches

            val_loss = None
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(val_tensor)
                    val_loss = criterion(val_output, val_tensor).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(val_loss, 6) if val_loss else None,
            })

            if (epoch + 1) % 10 == 0:
                msg = f"  Epoch {epoch + 1}/{self.epochs} | Train Loss: {avg_train_loss:.6f}"
                if val_loss:
                    msg += f" | Val Loss: {val_loss:.6f}"
                print(msg)

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                self.model.load_state_dict(self._best_state)
                break

        if X_val is not None:
            self._set_threshold(X_val)

        print(f"Training complete | Threshold: {self.threshold:.6f}")
        return self

    def _set_threshold(self, X_val):
        """Set anomaly threshold based on validation reconstruction errors.

        Parameters
        ----------
        X_val : array-like
            Validation data (should be normal operating data).
        """
        errors = self.compute_anomaly_scores(X_val)
        self.threshold = float(np.percentile(errors, self.threshold_percentile))
        print(f"  Threshold set at {self.threshold_percentile}th percentile: {self.threshold:.6f}")

    def compute_anomaly_scores(self, X):
        """Compute reconstruction error for each sample.

        Parameters
        ----------
        X : array-like
            Feature matrix (DataFrame or array).

        Returns
        -------
        numpy.ndarray
            Array of reconstruction error scores.
        """
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

        return errors.cpu().numpy()

    def predict(self, X):
        """Predict anomaly labels (0=normal, 1=anomaly).

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Array of binary predictions.
        """
        scores = self.compute_anomaly_scores(X)
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit() with validation data first.")
        return (scores > self.threshold).astype(int)

    def save(self, model_dir="models"):
        """Save the trained model, scaler, and threshold.

        Parameters
        ----------
        model_dir : str
            Directory to save model artifacts.
        """
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "training_history": self.training_history,
        }, path / "autoencoder.pth")

        joblib.dump(self.scaler, path / "autoencoder_scaler.pkl")
        print(f"Autoencoder saved to {path}")

    def load(self, model_dir="models"):
        """Load a trained model from disk.

        Parameters
        ----------
        model_dir : str
            Directory containing model artifacts.

        Returns
        -------
        GridAutoencoder
            self
        """
        path = Path(model_dir)
        checkpoint = torch.load(path / "autoencoder.pth", map_location=self.device)

        self.input_dim = checkpoint["input_dim"]
        self.threshold = checkpoint["threshold"]
        self.threshold_percentile = checkpoint["threshold_percentile"]
        self.training_history = checkpoint["training_history"]

        self.model = Autoencoder(self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.scaler = joblib.load(path / "autoencoder_scaler.pkl")

        print(f"Autoencoder loaded from {path} | Threshold: {self.threshold:.6f}")
        return self

    def get_model_info(self):
        """Return metadata about the trained model.

        Returns
        -------
        dict
            Model parameters and training info.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_type": "Autoencoder",
            "input_dim": self.input_dim,
            "architecture": "input->64->32->16->32->64->input",
            "total_parameters": total_params,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
            "epochs_trained": len(self.training_history),
            "device": str(self.device),
        }


if __name__ == "__main__":
    print("=== Autoencoder Demo ===")
    n_features = 30
    X_normal = np.random.randn(1000, n_features)
    X_anomaly = np.random.randn(50, n_features) * 3 + 2

    ae = GridAutoencoder(input_dim=n_features, epochs=20, patience=5)
    ae.fit(X_normal[:800], X_normal[800:])

    normal_scores = ae.compute_anomaly_scores(X_normal[:100])
    anomaly_scores = ae.compute_anomaly_scores(X_anomaly)
    print(f"Normal scores: mean={normal_scores.mean():.4f}")
    print(f"Anomaly scores: mean={anomaly_scores.mean():.4f}")
    print(ae.get_model_info())
