from typing import Tuple

import numpy as np

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from tensorflow_addons.utils.types import AcceptableDTypes
from tensorflow.python.ops import weights_broadcast_ops





class RSquare(metrics.Metric):
    def __init__(
        self,
        uncertainty: bool = False,
        name: str = "r_square",
        dtype: AcceptableDTypes = None,
        y_shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        super(RSquare, self).__init__(name=name, dtype=dtype, **kwargs)
        self.uncertainty = uncertainty
        self.y_shape = y_shape

        self.squared_sum = self.add_weight(
            name="squared_sum", shape=y_shape, initializer="zeros", dtype=dtype
        )
        self.sum = self.add_weight(
            name="sum", shape=y_shape, initializer="zeros", dtype=dtype
        )
        self.res = self.add_weight(
            name="residual", shape=y_shape, initializer="zeros", dtype=dtype
        )
        self.count = self.add_weight(
            name="count", shape=y_shape, initializer="zeros", dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_true = tf.cast(y_true, dtype=self._dtype)
        y_pred = tf.cast(y_pred, dtype=self._dtype)

        if self.uncertainty:
            y_pred = y_pred[:, ::2]


        if sample_weight is None:
            sample_weight = 1
        sample_weight = tf.cast(sample_weight, dtype=self._dtype)
        sample_weight = weights_broadcast_ops.broadcast_weights(
            weights=sample_weight, values=y_true
        )

        weighted_y_true = y_true * sample_weight
        self.sum.assign_add(tf.reduce_sum(weighted_y_true, axis=0))
        self.squared_sum.assign_add(tf.reduce_sum(y_true * weighted_y_true, axis=0))
        self.res.assign_add(
            tf.reduce_sum((y_true - y_pred) ** 2 * sample_weight, axis=0)
        )
        self.count.assign_add(tf.reduce_sum(sample_weight, axis=0))

    def result(self) -> tf.Tensor:
        mean = self.sum / self.count
        total = self.squared_sum - self.sum * mean
        raw_scores = 1 - (self.res / total)
        raw_scores = tf.where(tf.math.is_inf(raw_scores), 0.0, raw_scores)

        return raw_scores

    def reset_state(self) -> None:
        # The state of the metric will be reset at the start of each epoch.
        K.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])

    def get_config(self):
        config = {
            "y_shape": self.y_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}



class RootMeanSquaredError(metrics.Metric):
    def __init__(
        self,
        uncertainty: bool = False,
        name: str = "root_mean_squared_error",
        dtype: AcceptableDTypes = None,
        y_shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        super(RootMeanSquaredError, self).__init__(name, dtype=dtype)

        self.uncertainty = uncertainty
        self.y_shape = y_shape

        self.squared_sum = self.add_weight(
            name="squared_sum", shape=y_shape, initializer="zeros", dtype=dtype
        )
        self.count = self.add_weight(
            name="count", shape=y_shape, initializer="zeros", dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        if self.uncertainty:
            y_pred = y_pred[:, ::2]

        if sample_weight is None:
            sample_weight = 1
        sample_weight = tf.cast(sample_weight, dtype=self._dtype)
        sample_weight = weights_broadcast_ops.broadcast_weights(
            weights=sample_weight, values=y_true
        )

        self.squared_sum.assign_add(
            tf.reduce_sum((y_true - y_pred) ** 2 * sample_weight, axis=0)
        )
        self.count.assign_add(tf.reduce_sum(sample_weight, axis=0))

    
    def result(self) -> tf.Tensor:
        return tf.sqrt(tf.math.divide_no_nan(self.squared_sum, self.count))

    def reset_state(self) -> None:
        # The state of the metric will be reset at the start of each epoch.
        K.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])

    def get_config(self):
        config = {
            "y_shape": self.y_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}



class MeanAbsoluteError(metrics.Metric):  
    def __init__(
        self,
        uncertainty: bool = False,
        name: str = "mean_absolte_error",
        dtype: AcceptableDTypes = None,
        y_shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        super(MeanAbsoluteError, self).__init__(name, dtype=dtype)

        self.uncertainty = uncertainty
        self.y_shape = y_shape

        self.absolte_sum = self.add_weight(
            name="absolte_sum", shape=y_shape, initializer="zeros", dtype=dtype
        )
        self.count = self.add_weight(
            name="count", shape=y_shape, initializer="zeros", dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        if self.uncertainty:
            y_pred = y_pred[:, ::2]

        if sample_weight is None:
            sample_weight = 1
        sample_weight = tf.cast(sample_weight, dtype=self._dtype)
        sample_weight = weights_broadcast_ops.broadcast_weights(
            weights=sample_weight, values=y_true
        )

        self.absolte_sum.assign_add(
            tf.reduce_sum(tf.abs(y_true - y_pred) * sample_weight, axis=0)
        )
        self.count.assign_add(tf.reduce_sum(sample_weight, axis=0))

    
    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.absolte_sum, self.count)

    def reset_state(self) -> None:
        # The state of the metric will be reset at the start of each epoch.
        K.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])

    def get_config(self):
        config = {
            "y_shape": self.y_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}



class STD(metrics.Metric):  
    def __init__(
        self,
        uncertainty: bool = False,
        name: str = "std",
        dtype: AcceptableDTypes = None,
        y_shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        super(STD, self).__init__(name, dtype=dtype)
        self.uncertainty = uncertainty
        self.y_shape = y_shape

        self.sum_std = self.add_weight(
            name="sum_std", shape=y_shape, initializer="zeros", dtype=dtype
        )
        self.count = self.add_weight(
            name="count", shape=y_shape, initializer="zeros", dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        if self.uncertainty:
            y_pred = y_pred[:, 1::2]

        if sample_weight is None:
            sample_weight = 1
        sample_weight = tf.cast(sample_weight, dtype=self._dtype)
        sample_weight = weights_broadcast_ops.broadcast_weights(
            weights=sample_weight, values=y_true
        )

        self.sum_std.assign_add(
            tf.reduce_sum(y_pred * sample_weight, axis=0)
        )
        self.count.assign_add(tf.reduce_sum(sample_weight, axis=0))

    
    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.sum_std, self.count)

    def reset_state(self) -> None:
        # The state of the metric will be reset at the start of each epoch.
        K.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])

    def get_config(self):
        config = {
            "y_shape": self.y_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}