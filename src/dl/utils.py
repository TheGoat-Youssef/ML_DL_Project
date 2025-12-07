import tensorflow as tf
import numpy as np

def to_categorical_if_needed(y, num_classes=None):
    """Convert y to one-hot if it's integer labels and user wants one-hot."""
    if y.ndim == 1:
        if num_classes is None:
            num_classes = int(y.max()) + 1
        return tf.keras.utils.to_categorical(y, num_classes)
    return y

def make_tf_dataset(X, y, batch_size=32, shuffle=True, buffer_size=1024, one_hot=False):
    """
    Create a tf.data.Dataset from numpy arrays.
    If one_hot True, y will be converted to categorical.
    """
    if one_hot:
        num_classes = int(y.max()) + 1
        y = tf.keras.utils.to_categorical(y, num_classes)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
