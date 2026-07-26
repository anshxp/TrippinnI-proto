"""
Autoencoder.

Deep learning model for anomaly detection.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 16,
    ) -> None:

        inputs = Input(shape=(input_dim,))

        encoded = Dense(
            encoding_dim,
            activation="relu",
        )(inputs)

        decoded = Dense(
            input_dim,
            activation="linear",
        )(encoded)

        self.model = Model(inputs, decoded)

        self.model.compile(
            optimizer="adam",
            loss=MeanSquaredError(),
        )

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> None:
        """
        Train the autoencoder.
        """

        self.model.fit(
            X,
            X,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def reconstruction_error(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute reconstruction error.
        """

        reconstructed = self.model.predict(
            X,
            verbose=0,
        )

        return np.mean(
            np.square(X - reconstructed),
            axis=1,
        )

    def predict(
        self,
        X: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Predict anomalies.

        Returns
        -------
        ndarray
            0 -> Normal
            1 -> Outlier
        """

        errors = self.reconstruction_error(X)

        return (errors > threshold).astype(int)

    def fit_predict(
        self,
        X: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Train and predict.
        """

        self.fit(X)

        return self.predict(
            X,
            threshold,
        )