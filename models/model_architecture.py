import os
import json

def get_class_labels(labels_path=None):
    if labels_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        labels_path = os.path.join(base_dir, 'labels.json')
    with open(labels_path, 'r') as f:
        return json.load(f)

def build_mobilenetv2_model(num_classes=38, input_shape=(224, 224, 3), weights='imagenet'):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2

    inputs = Input(shape=input_shape, name='plant_image_input')
    base_model = MobileNetV2(input_tensor=inputs, include_top=False, weights=weights)
    base_model.trainable = False

    x = GlobalAveragePooling2D(name='global_avg_pool')(base_model.output)
    x = BatchNormalization(name='bn_pool')(x)
    x = Dropout(0.35, name='dropout_1')(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4), name='dense_features')(x)
    x = BatchNormalization(name='bn_dense')(x)
    x = Dropout(0.25, name='dropout_2')(x)

    outputs = Dense(num_classes, activation='softmax', name='disease_predictions')(x)
    model = Model(inputs=inputs, outputs=outputs, name='PlantVillage_MobileNetV2')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
