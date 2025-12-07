import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters, kernel=3, pool=True):
    x = layers.Conv2D(filters, kernel, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if pool:
        x = layers.MaxPooling2D(2)(x)
    return x

def build_simple_cnn(input_shape, num_classes, base_filters=32, dropout=0.4):
    inputs = layers.Input(shape=input_shape)
    x = conv_block(inputs, base_filters)
    x = conv_block(x, base_filters*2)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="cnn_simple")

def build_deeper_cnn(input_shape, num_classes, base_filters=32, dropout=0.5):
    inputs = layers.Input(shape=input_shape)
    x = conv_block(inputs, base_filters)
    x = conv_block(x, base_filters*2)
    x = conv_block(x, base_filters*4)
    x = layers.Conv2D(base_filters*8, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="cnn_deeper")
