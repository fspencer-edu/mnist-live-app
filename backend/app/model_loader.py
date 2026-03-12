from __future__ import annotations

from functools import lru_cache
import tensorflow as tf

MODEL_PATH = "../models/mnist_cnn.keras"

@lru_cache(maxsize=1)
def get_model() -> tf.keras.Model:
    model = tf.keras.models.load_model(MODEL_PATH)
    return model