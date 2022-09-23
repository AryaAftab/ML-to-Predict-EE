import math

import tensorflow as tf
from tensorflow.keras import losses



class GaussLoss(losses.Loss):
    """Negative log likelihood of y_true, with the likelihood defined by a normal distribution."""
    def __init__(self, name="loss"):
        super(GaussLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        means = y_pred[:, ::2]
        # We predict the log of the standard deviation, so exponentiate the prediction here
        stds = tf.exp(y_pred[:, 1::2])
        variances = stds * stds


        log_p = (-tf.math.log(tf.math.sqrt(2 * math.pi * variances) + 1e-8)
                 -(y_true - means) * (y_true - means) / (2 * variances))
        return tf.reduce_mean(-log_p, axis=-1)  # Note the `axis=-1`



class MAEM(losses.Loss):
	def __init__(self, epsilon=0.05, name="loss", **kwargs):
		super(MAEM, self).__init__(name=name)
		self.epsilon = epsilon

	def call(self, y_true, y_pred):
		difference = tf.maximum(tf.abs(y_true - y_pred) - self.epsilon, 0.0)
		return tf.reduce_mean(difference, axis=-1)