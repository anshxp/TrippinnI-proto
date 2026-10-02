"""Optional TensorFlow reconstruction learner with lazy import."""
from __future__ import annotations
import numpy as np

def reconstruction_errors(matrix: np.ndarray, epochs: int=15, random_state: int=42) -> np.ndarray:
    X=np.asarray(matrix,dtype=np.float32)
    if X.ndim!=2 or len(X)<50: return np.zeros(len(X))
    from sklearn.preprocessing import RobustScaler
    Z=RobustScaler().fit_transform(X).astype(np.float32)
    try:
        import tensorflow as tf
        tf.random.set_seed(random_state)
        inputs=tf.keras.Input(shape=(Z.shape[1],))
        encoded=tf.keras.layers.Dense(max(2,min(32,Z.shape[1]*2)),activation="relu")(inputs)
        decoded=tf.keras.layers.Dense(Z.shape[1])(encoded)
        model=tf.keras.Model(inputs,decoded)
        model.compile(optimizer="adam",loss="mse")
        model.fit(Z,Z,epochs=epochs,batch_size=min(256,len(Z)),verbose=0)
        return np.mean((Z-model.predict(Z,verbose=0))**2,axis=1)
    except Exception:
        return np.zeros(len(X))
