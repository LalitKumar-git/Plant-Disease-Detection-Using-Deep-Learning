
import os
import json
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)


from models.model_architecture import build_mobilenetv2_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")


def normalize_label(label):
    label = label.replace(
        "Cherry___",
        "Cherry_(including_sour)___"
    )

    label = label.replace(
        "Corn___",
        "Corn_(maize)___"
    )

    label = label.replace(
        "Corn_(maize)___Common_rust",
        "Corn_(maize)___Common_rust_"
    )

    return label


def train(epochs=8, batch_size=16,
          output_weights="models/plant_disease_mobilenetv2.h5"):

    print("\nLoading project labels...")

    with open(LABELS_PATH, "r") as f:
        project_labels = json.load(f)

    project_label_to_index = {
        label: index
        for index, label in enumerate(project_labels)
    }

    print("Loading PlantVillage dataset...")

    (train_ds, val_ds), info = tfds.load(
        "plant_village",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True
    )

    tfds_labels = info.features["label"].names

    label_mapping = []

    for label in tfds_labels:
        normalized = normalize_label(label)

        if normalized not in project_label_to_index:
            raise ValueError(
                f"Dataset label not found in labels.json: {label}"
            )

        label_mapping.append(
            project_label_to_index[normalized]
        )

    if len(tfds_labels) != len(project_labels):
        raise ValueError("Dataset and project class counts do not match.")

    print("All 38 classes mapped successfully!")

    mapping_tensor = tf.constant(label_mapping, dtype=tf.int32)
    num_classes = len(project_labels)
    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224))

        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image
        )

        label = tf.gather(mapping_tensor, label)
        label = tf.one_hot(label, num_classes)

        return image, label

    print("Preparing training data...")

    train_ds = (
        train_ds
        .shuffle(512)
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    print("\nBuilding MobileNetV2 model...")

    model = build_mobilenetv2_model(
        num_classes=num_classes,
        weights="imagenet"
    )

    os.makedirs(
        os.path.dirname(output_weights) or ".",
        exist_ok=True
    )

    callbacks = [
        ModelCheckpoint(
            output_weights,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            verbose=1
        )
    ]

    print("\nStarting model training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    model.save(output_weights)

    print("\nTraining completed!")
    print("Model saved at:", output_weights)

    print(
        "Final training accuracy:",
        round(history.history["accuracy"][-1] * 100, 2),
        "%"
    )

    print(
        "Final validation accuracy:",
        round(history.history["val_accuracy"][-1] * 100, 2),
        "%"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=8
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--output_weights",
        default="models/plant_disease_mobilenetv2.h5"
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_weights=args.output_weights
    )
