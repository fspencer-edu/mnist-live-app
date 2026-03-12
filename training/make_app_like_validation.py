import tensorflow as tf

def make_app_like_test_set(x_test: tf.Tensor) -> tf.Tensor:
    transform = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(factor=0.12),
            tf.keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15),
            tf.keras.layesr.RandomTranslation(
                height_factor=0.15,
                width_factor=0.15,
                fill_mode="constant",
                fill_value=0.0,
            ),
            tf.keras.layers.RandomConstrast(factor=0.15),
        ]
    )
    return transform(x_test, training=True)