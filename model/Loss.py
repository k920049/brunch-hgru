import tensorflow as tf

from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.ops.nn_impl import *
from tensorflow.python.keras.losses import *
from tensorflow.python.keras.engine.base_layer import Layer


class RankingLoss(Layer):

    def __init__(self,
                 num_units,
                 num_sampled,
                 num_classes,
                 num_true=1,
                 history_length=30,
                 sampled_values=None,
                 remove_accidental_hits=False,
                 partition_strategy="mod",
                 **kwargs):

        self.num_units = num_units
        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.num_true = num_true
        self.history_length = history_length
        self.sampled_values = sampled_values
        self.remove_accidental_hits = remove_accidental_hits
        self.partition_strategy = partition_strategy

        super(RankingLoss, self).__init__(kwargs)

    def build(self, input_shape):

        self.w = self.add_weight(
            name="approx_softmax_weights",
            shape=(self.num_classes + 1, self.num_units),
            initializer="glorot_normal",
        )

        self.b = self.add_weight(
            name="approx_softmax_biases", shape=(self.num_classes + 1,),
            initializer="zeros"
        )

        # keras
        super(RankingLoss, self).build(input_shape)

    def call(self, inputs):
        labels = tf.gather(inputs, [self.num_units], axis=1)
        inputs = tf.gather(inputs, [i for i in range(self.num_units)], axis=1)
        return self.ranking_loss(labels=labels, inputs=inputs)

    def ranking_loss(self, labels, inputs):

        logits, labels = _compute_sampled_logits(
            weights=self.w,
            biases=self.b,
            labels=labels,
            inputs=inputs,
            num_sampled=self.num_sampled,
            num_classes=self.num_classes,
            num_true=self.num_true,
            sampled_values=self.sampled_values,
            subtract_log_q=True,
            remove_accidental_hits=self.remove_accidental_hits,
            partition_strategy=self.partition_strategy)
        # prevent backpropagation through labels
        labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")
        # divided true score from sampled score
        true_logit = tf.gather(logits, [0], axis=1)
        sampled_logit = tf.gather(logits, [i + 1 for i in range(self.num_sampled)], axis=1)
        # compute top1 loss
        loss = tf.math.sigmoid(sampled_logit - true_logit) + tf.math.sigmoid(tf.square(sampled_logit))
        loss = tf.reduce_mean(loss, axis=1, keepdims=True)
        # predicted score
        predict = tf.math.sigmoid(true_logit)
        return tf.concat([loss, predict], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2)

    def get_config(self):
        elems = {
            'num_units': self.num_units,
            'num_sampled': self.num_sampled,
            'num_classes': self.num_classes,
            'num_true': self.num_true,
            'history_length': self.history_length,
            'sampled_values': self.sampled_values,
            'remove_accidental_hits': self.remove_accidental_hits,
            'partition_strategy': self.partition_strategy
        }
        config = {
            'ranking_loss': elems
        }
        base_config = super(RankingLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config.pop('ranking_loss'))
