
import os
import json
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(BASE_DIR, 'plant_disease_mobilenetv2.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.json')


def get_labels():
    with open(LABELS_PATH, 'r') as f:
        return json.load(f)


def get_weights_location_info():
    return {
        "expected_path": os.path.abspath(DEFAULT_WEIGHTS_PATH),
        "exists": os.path.exists(DEFAULT_WEIGHTS_PATH),
        "instructions": "Place your trained weights file at: models/plant_disease_mobilenetv2.h5"
    }


def preprocess_image(image_input, target_size=(224, 224)):
    img = Image.open(image_input).convert('RGB')
    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_preprocessed = (img_array / 127.5) - 1.0
    return np.expand_dims(img_preprocessed, axis=0), img


def run_cnn_inference(image_input):
    import tensorflow as tf

    labels = get_labels()
    batch_tensor, _ = preprocess_image(image_input)

    if not os.path.exists(DEFAULT_WEIGHTS_PATH):
        raise FileNotFoundError(
            "Trained model not found. Please train the model first."
        )

    model = tf.keras.models.load_model(
        DEFAULT_WEIGHTS_PATH,
        compile=False
    )

    preds = model.predict(batch_tensor, verbose=0)[0]

    if len(preds) != len(labels):
        raise ValueError("Model output classes do not match labels.json")

    top_indices = np.argsort(preds)[::-1]
    top_idx = top_indices[0]
    top_class = labels[top_idx]

    crop, disease = (
        top_class.split('___', 1)
        if '___' in top_class
        else ('Plant', top_class)
    )

    top_5 = []

    for idx in top_indices[:5]:
        c, d = (
            labels[idx].split('___', 1)
            if '___' in labels[idx]
            else ('Plant', labels[idx])
        )

        top_5.append({
            "class_id": labels[idx],
            "crop": c.replace('_', ' '),
            "disease": d.replace('_', ' '),
            "confidence": round(float(preds[idx] * 100), 2)
        })

    return {
        "predicted_class": top_class,
        "crop": crop.replace('_', ' '),
        "disease": disease.replace('_', ' '),
        "confidence": round(float(preds[top_idx] * 100), 2),
        "is_healthy": "healthy" in top_class.lower(),
        "top_5": top_5,
        "weights_loaded": True
    }
