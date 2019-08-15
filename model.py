import tensorflow as tf
import pandas as pd
import numpy as np
import datetime

from model.Wrapper import HierarchicalRNNCell
from model.Loss import RankingLoss
from pipeline.Datasets import Datasets

from tensorflow.python.keras.layers.recurrent import RNN, GRUCell
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.wrappers import TimeDistributed
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.metrics import BinaryAccuracy
from tensorflow.python import debug as tf_debug
from tensorflow.python.keras import backend as K


def custom_loss(y_true, y_pred):
    return tf.reduce_sum(y_pred)

def custom_acc(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=2)
    y_pred = tf.squeeze(y_pred, axis=2)

    num_item = tf.math.reduce_sum(y_true, axis=1, keepdims=True)
    accuracy = y_true - y_pred
    accuracy = tf.math.reduce_sum(accuracy, axis=1, keepdims=True)
    accuracy = tf.divide(accuracy, num_item)

    return 1.0 - tf.math.reduce_mean(accuracy)


class HierarchicalRecommender(object):

    def __init__(self,
                 history_length=30,
                 num_negatives=4,
                 num_units=256,
                 batch_size=256,
                 epoch=10):
        self.history_length = history_length
        self.num_negatives = num_negatives
        self.num_units = num_units
        self.batch_size = batch_size
        self.epoch = epoch

        self.embedding_mx = np.load("./data/brunch/embedding.npy")
        self.embedding_mx = np.concatenate([self.embedding_mx, np.zeros((1, self.embedding_mx.shape[1]))], axis=0)
        self.vocab_size = self.embedding_mx.shape[0]
        self.embedding_dim = self.embedding_mx.shape[1]

        self.model = self._build_model()

    def _build_model(self):
        with tf.name_scope("inputs"):
            user_input = tf.keras.Input(shape=(self.history_length, 3))
            label_input = tf.keras.Input(shape=(self.history_length, 1))
            mask_input = tf.keras.Input(shape=(self.history_length, 1))

        with tf.name_scope("layers"):
            embedding = Embedding(input_dim=self.vocab_size,
                                  output_dim=self.embedding_dim,
                                  weights=[self.embedding_mx],
                                  trainable=False)
            session_cells = [
                GRUCell(units=self.num_units, name="sesion_rnn_01"),
                GRUCell(units=self.num_units, name="sesion_rnn_02")
                # GRUCell(units=self.num_units, name="sesion_rnn_03")
            ]
            user_cells = [
                GRUCell(units=self.num_units, name="user_rnn_01"),
                GRUCell(units=self.num_units, name="user_rnn_02")
                # GRUCell(units=self.num_units, name="user_rnn_03")
            ]
            cell = HierarchicalRNNCell(user_cells=user_cells,
                                       session_cells=session_cells,
                                       embedding_layer=embedding)
            recurrent = RNN(cell=cell,
                            return_sequences=True,
                            return_state=True)

        with tf.name_scope("loss"):

            loss = RankingLoss(num_units=self.num_units,
                               num_sampled=self.num_negatives,
                               num_classes=self.vocab_size - 1,
                               num_true=1,
                               history_length=self.history_length,
                               remove_accidental_hits=True)

            time_distributed = TimeDistributed(loss, input_shape=(self.history_length, self.num_units + 1))

        with tf.name_scope("model"):
            tensor = recurrent(inputs=user_input)
            outputs = tensor[0]
            outputs = tf.concat([outputs, label_input], axis=2)
            tensor = time_distributed(outputs)
            # loss
            loss = tf.gather(tensor, [0], axis=2)
            loss = tf.multiply(loss, mask_input, name="loss")
            # prediction
            prediction = tf.gather(tensor, [1], axis=2)
            prediction = tf.multiply(prediction, mask_input, name="prediction")
            # build the model
            model = tf.keras.Model(inputs=[user_input, label_input, mask_input], outputs=[loss, prediction])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss={'tf_op_layer_loss': custom_loss,
                                'tf_op_layer_prediction': 'binary_crossentropy'},
                          loss_weights={'tf_op_layer_loss': 1.0,
                                'tf_op_layer_prediction': 0.0},
                          metrics={'tf_op_layer_prediction': custom_acc})
        return model

    def train(self):
        dataset = Datasets("./data/brunch")
        data, label, mask = dataset.read()
        print(data.shape, label.shape, mask.shape)
        # save model every 5 epoch
        filepath = "./data/checkpoints/init/model-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     save_weights_only=False,
                                     verbose=1)
        # display tensorboard
        log_dir = "./data/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(x=[data, label, mask],
                       y=[mask, mask],
                       steps_per_epoch=None,
                       batch_size=self.batch_size,
                       shuffle=True,
                       epochs=10,
                       validation_split=0.1,
                       callbacks=[checkpoint, tensorboard_callback])


if __name__ == "__main__":
    HierarchicalRecommender().train()
