import tensorflow as tf

def build_augmentation () -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(factor=0.08),
            tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10),
            tf.keras.layers.RandomTranslation(
                height_factor=0.10,
                width_factor=0.10,
                fill_node="constant",
                fill_value=0.0,
            ),
            tf.keras.layers.RandomConstrast(factor=0.10),
        ],
        name="mnist_augmentation",
    )