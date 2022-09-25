import tensorflow as tf
from tensorflow.python.keras.backend import dtype
import numpy as np

def normalize(inputs):
    """Normalize the input image.

    Args:
        inputs: a TensorFlow tensor of image.

    Returns:
        a normalized image tensor.
    """
    inputs = tf.cast(inputs, dtype=tf.float32)
    img_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    img_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    return ((inputs / 255.0) - img_mean)/img_std


def normalize_numpy(inputs):

    
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])

    return ((inputs / 255.0) - img_mean)/img_std
