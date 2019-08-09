import tensorflow as tf
import pandas as pd
import numpy as np

from model.Wrapper import HierarchicalRNNCell
from model.Loss import RankingLoss
from pipeline.Datasets import Datasets

from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.cudnn_recurrent import CuDNNGRU

history_length = 30
num_negatives = 4
embedding_mx = np.load("./data/brunch/train/embedding.npy")
vocab_size = embedding_mx.shape[0],
embedding_dim = embedding_mx.shape[1]
num_units = 256
batch_size = 256
epoch = 10


def main():

    with tf.name_scope("inputs"):
        user_input = tf.keras.Input(shape=(history_length, 3))
        item_input = tf.keras.Input(shape=(history_length, 1))
        mask_input = tf.keras.Input(shape=(history_length, 1))

    with tf.name_scope("layers"):
        embedding = Embedding(input_dim=vocab_size,
                              output_dim=embedding_dim,
                              weights=embedding_mx,
                              trainable=False)
        session_cells = [
            CuDNNGRU(units=num_units,
                     return_sequences=True,
                     return_state=True),
            CuDNNGRU(units=num_units,
                     return_sequences=True,
                     return_state=True),
            CuDNNGRU(units=num_units,
                     return_sequences=True,
                     return_state=True)
        ]
        user_cells = [
            CuDNNGRU(units=num_units,
                     return_sequences=True,
                     return_state=True),
            CuDNNGRU(units=num_units,
                     return_sequences=True,
                     return_state=True),
            CuDNNGRU(units=num_units,
                     return_sequences=True,
                     return_state=True)
        ]
        cell = HierarchicalRNNCell(user_cells=user_cells,
                                   session_cells=session_cells,
                                   embedding_layer=embedding)
        recurrent = RNN(cell=cell,
                        return_sequences=True,
                        return_state=True)

    with tf.name_scope("loss"):
        nce_weights = tf.Variable(
            tf.random.truncated_normal([vocab_size, embedding_dim], stddev=1.0 / np.sqrt(embedding_dim)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))
        loss = RankingLoss(weights=nce_weights,
                           biases=nce_biases,
                           num_sampled=num_negatives,
                           num_classes=vocab_size,
                           num_true=1,
                           history_length=history_length)


    with tf.name_scope("model"):
        initial_state = [tf.zeros(shape=(tf.shape(user_input)[0], num_units))] * 6
        tensor = recurrent(inputs=user_input, initial_state=initial_state)              # [batch size][time step][output dimension]
        tensor = tf.concat([tensor, mask_input], axis=2)                                # [batch size][time step][output dimension + 1]
        model = tf.keras.Model(inputs=[user_input, mask_input], outputs=tensor)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=loss,
                      metrics=['accuracy'])

    with tf.name_scope("train"):
        dataset = Datasets("../data/brunch/train")
        data, label, mask = dataset.read()
        filepath = "../data/checkpoints/init/model-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     save_weights_only=True,
                                     verbose=1)
        self.model.fit(x = [data, mask],
                       y = label,
                       batch_size=batch_size,
                       shuffle=True,
                       epochs=epoch,
                       callbacks=[checkpoint])


if __name__ == "__main__":
    main()
