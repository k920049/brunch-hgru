import tensorflow as tf
import pandas as pd
import numpy as np

from model.Wrapper import HierarchicalRNNCell
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.cudnn_recurrent import CuDNNGRU

history_length = 32
num_negatives = 4
embedding_mx = None
vocab_size = 1234,
embedding_dim = 300
num_units = 256

def main():

    with tf.name_scope("inputs"):
        user_input = tf.keras.Input(shape=(history_length, ))
        item_input = tf.keras.Input(shape=(num_negatives + 1, ))

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

    with tf.name_scope("model"):
        initial_state = [tf.zeros(shape=(tf.shape(user_input)[0], num_units))] * 6
        tensor = recurrent(inputs=user_input, initial_state=initial_state)
        loss = tf.nn.sampled_softmax_loss(weights=embedding_mx,
                                          biases=tf.zeros(vocab_size, ),
                                          labels=item_input,
                                          inputs=tensor,
                                          num_sampled=num_negatives,
                                          num_classes=vocab_size,
                                          partition_strategy='div')
        model = tf.keras.Model(inputs=[user_input, item_input], outputs=tensor)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=loss,
                      metrics=['accuracy'])



if __name__ == "__main__":
    main()