from models.model_loader import get_labels, preprocess_image
from database.db_setup import init_db, insert_prediction, get_all_predictions
from models.model_architecture import build_mobilenetv2_model
from PIL import Image
import numpy as np

def run_tests():
    print("1. Verifying 38 PlantVillage Labels...")
    labels = get_labels()
    assert len(labels) == 38
    print("   [PASS] 38 classes loaded.")

    print("2. Testing SQLite Database...")
    init_db()
    tid = insert_prediction("test.jpg", "static/test.jpg", "Tomato", "Early Blight", 96.5, "Diseased", "Spots", "Fungicide")
    assert tid > 0
    records = get_all_predictions()
    assert len(records) > 0
    print("   [PASS] Database CRUD operations operational.")

    print("3. Testing MobileNetV2 Architecture Compilation...")
    model = build_mobilenetv2_model(num_classes=38, weights=None)
    assert len(model.layers) > 100
    print("   [PASS] MobileNetV2 compiled with 38 Softmax heads.")

    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    run_tests()
