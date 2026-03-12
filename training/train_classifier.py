from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from augment import build_augmentation
from make_app_like_validation import make_app_like_test_set
from metrics import save_history


SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
OUTPUT_DIR = Path("outputs")
MODEL_PATH = OUTPUT_DIR / "mnist_cnn.keras"
CLASS_NAMES_PATH = OUTPUT_DIR / "class_names.json"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # shape: (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    augmentation = build_augmentation()

    inputs = tf.keras.Input(shape=(28, 28, 1), name="image")
    x = augmentation(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(10, activation="softmax", name="probabilities")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.keras.utils.set_random_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"MNIST test accuracy: {test_acc:.4f}")

    x_test_app_like = make_app_like_test_set(tf.convert_to_tensor(x_test))
    app_like_loss, app_like_acc = model.evaluate(x_test_app_like, y_test, verbose=0)
    print(f"App-like test accuracy: {app_like_acc:.4f}")

    model.save(MODEL_PATH)
    save_history(history, str(HISTORY_PATH))

    class_names = {str(i): str(i) for i in range(10)}
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()