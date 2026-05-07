"""
model_utils.py
--------------
Defines the CNN model using transfer learning.
Supports: EfficientNetB0 (default), MobileNetV2, ResNet50.
"""

import tensorflow as tf
from keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SHAPE = (224, 224, 3)


def build_model(backbone: str = "mobilenet",
                dropout_rate: float = 0.4,
                l2_reg: float = 1e-4,
                learning_rate: float = 1e-4) -> tf.keras.Model:
    """
    Build a binary classifier on top of a pretrained backbone.

    Phase 1 – base frozen   → train only the head
    Phase 2 – fine-tune     → unfreeze top N layers of base

    Returns a compiled model ready for Phase-1 training.
    """
    # ── Select backbone ──────────────────────────────────────────────────────
    backbone_map = {
        "efficientnet": EfficientNetB0,
        "mobilenet":    MobileNetV2,
        "resnet50":     ResNet50,
    }
    assert backbone in backbone_map, f"Unknown backbone: {backbone}"
    BaseModel = backbone_map[backbone]

    base_model = BaseModel(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False          # Phase 1: frozen

    # ── Classification head ──────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)   # binary

    model = models.Model(inputs, outputs, name=f"BreastCancer_{backbone}")

    _compile(model, learning_rate)
    return model


def unfreeze_top_layers(model: tf.keras.Model,
                        n_layers: int = 20,
                        fine_tune_lr: float = 1e-5) -> tf.keras.Model:
    """
    Unfreeze the top `n_layers` of the base model for fine-tuning.
    Call after Phase-1 training converges.
    """
    base = next(layer for layer in model.layers if hasattr(layer, "layers"))          # second layer is always the backbone
    base.trainable = True

    # Freeze everything except the last n_layers
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    _compile(model, fine_tune_lr)
    print(f"[model_utils] Unfroze top {n_layers} base layers — "
          f"LR set to {fine_tune_lr}")
    return model


def _compile(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
        ],
    )


def model_summary(model: tf.keras.Model) -> None:
    model.summary()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    total     = sum(tf.size(w).numpy() for w in model.weights)
    print(f"\nTrainable params : {trainable:,}")
    print(f"Total params     : {total:,}")
