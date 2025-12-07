import tensorflow as tf
from tensorflow.keras import layers, models

def build_mlp(input_shape, num_classes, hidden_units=(512, 256), dropout=0.4):
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    for units in hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="mlp_baseline")
    return model
