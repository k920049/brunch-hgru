import tensorflow as tf

from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.ops.nn_impl import *
from tensorflow.python.keras.losses import *


class RankingLoss(Loss):

    def __init__(self,
                 weights,
                 biases,
                 num_sampled,
                 num_classes,
                 num_true=1,
                 history_length=30,
                 sampled_values=None,
                 remove_accidental_hits=False,
                 partition_strategy="mod",
                 reduction=losses_utils.ReductionV2.AUTO,
                 name="ranking_loss",
                 **kwargs):
        self.weights = weights
        self.biases = biases
        self.num_sampled = num_sampled,
        self.num_classes = num_classes
        self.num_true = num_true
        self.sampled_values = sampled_values
        self.remove_accidental_hits = remove_accidental_hits
        self.partition_strategy = partition_strategy
        self._fn_kwargs = kwargs

        super(RankingLoss, self).__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        return self.loss_function(y_true, y_pred)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
        base_config = super(RankingLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def loss_function(self, y_true, y_pred):
        mask = y_true[:, :, -1]
        y_true = y_true[:, :, :-1]
        tensor_list = []

        for idx in range(self.history_length):
            step = tf.squeeze(tf.gather(y_pred, [idx], axis=1))
            label = tf.squeeze(tf.gather(y_true, [idx], axis=1))

            loss = ranking_loss(labels=label,
                                inputs=step)
            tensor_list.append(loss)
        tensor = tf.concat(tensor_list, axis=0)                 # [time step][batch size][loss = 1]
        tensor = tf.transpose(tensor, perm=[1, 0, 2])           # [batch_size][time_step][loss = 1]
        tensor_masked = tf.boolean_mask(tensor, mask, axis=1)   # [batch_size][variable length time step][loss]
        tensor_masked = tf.squeeze(tensor_masked, axis=-1)      # [batch size][variable length time step]
        return tf.reduce_mean(tensor_masked, axis=1)            # [batch_size, ]


    def ranking_loss(self, labels, inputs):

        logits, labels = _compute_sampled_logits(
            weights=self.weights,
            biases=self.biases,
            labels=labels,
            inputs=inputs,
            num_sampled=self.num_sampled,
            num_classes=self.num_classes,
            num_true=self.num_true,
            sampled_values=self.sampled_values,
            subtract_log_q=True,
            remove_accidental_hits=self.remove_accidental_hits,
            partition_strategy=self.partition_strategy,
            name=self.name)

        true_logit = tf.gather(logits, [0], axis=1)
        sampled_logit = tf.gather(logits, [i + 1 for i in range(num_sampled)], axis=1)
        loss = tf.math.sigmoid(sampled_logit - true_logit) + tf.math.sigmoid(tf.square(sampled_logit))

        return tf.reduce_mean(loss, axis=1, keepdims=True)


