from __future__ import annotations

import numpy as np
import tensorflow as tf


def compute_synflow(model: tf.keras.Model, input_shape=(64, 64, 3)) -> float:
    """
    TensorFlow/Keras adaptation of SynFlow.

    Steps:
    1) linearize weights -> abs(weights)
    2) forward with input of ones
    3) compute gradients of sum(output) w.r.t. weights
    4) accumulate sum(abs(w * grad))
    5) restore original signs
    """
    if not model.built:
        _ = model(tf.ones((1,) + input_shape, dtype=tf.float32), training=False)

    weight_vars = []
    for layer in model.layers:
        for var in layer.trainable_variables:
            name = var.name.lower()
            if "bias" in name:
                continue
            weight_vars.append(var)

    if not weight_vars:
        return 0.0

    signs = [np.sign(w.numpy()) for w in weight_vars]

    for w in weight_vars:
        w.assign(tf.abs(w))

    try:
        x = tf.ones((1,) + input_shape, dtype=tf.float32)

        with tf.GradientTape() as tape:
            y = model(x, training=False)
            loss = tf.reduce_sum(y)

        grads = tape.gradient(loss, weight_vars)

        score = 0.0
        for w, g in zip(weight_vars, grads):
            if g is not None:
                score += float(tf.reduce_sum(tf.abs(w * g)).numpy())

    finally:
        for w, s in zip(weight_vars, signs):
            w.assign(w * s)

    return score