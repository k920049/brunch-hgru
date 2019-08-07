from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import json

from tqdm import tqdm
from model.sequence.Encoder import Encoder
from model.sequence.Attention import Attention
from handler.dataset import generate_train_dataset, generate_embedding, generate_evaluation_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from fire import Fire

class Recommender(object):

    def __init__(self,
                 epoch=5,
                 batch_size = 256,
                 evaluation_ratio = 0.1,
                 encoder_units = 256,
                 history_length=15):

        self.epoch = epoch
        self.batch_size = batch_size
        self.evaluation_ratio = evaluation_ratio
        self.encoder_units = encoder_units
        self.history_length = history_length

        self.model = self._build_model()

    def _build_model(self):

        self.embedding_mx = generate_embedding(path_to_dictionary="./data/positional_dictionary.json",
                                               path_to_embedding="./data/embedding.npz.npy")

        self.vocab_size = self.embedding_mx.shape[0]
        self.embedding_dim = self.embedding_mx.shape[1]

        with tf.name_scope("data"):
            user_input = tf.keras.Input(shape=(self.history_length,))
            item_input = tf.keras.Input(shape=(1,))

        with tf.name_scope("model"):
            # Sequencial Model
            encoder = Encoder(vocab_size=self.vocab_size,
                              embedding_dim=self.embedding_dim,
                              enc_units=self.encoder_units,
                              batch_size=self.batch_size,
                              embedding_mx=self.embedding_mx)
            sample_output, sample_hidden = encoder(user_input)
            # Attention layer
            attention_layer = Attention(units=10, history=self.history_length)
            attension_result, attention_weights = attention_layer(sample_hidden, sample_output)
            # user dense layer
            #user = tf.keras.layers.Dense(units=512, activation="relu")(attension_result)
            user = tf.keras.layers.Dense(units=256, activation="relu")(attension_result)
            user = tf.keras.layers.Dense(units=128, activation="relu")(attension_result)
            user = tf.keras.layers.Dense(units=64, activation="relu")(user)
            # user = tf.keras.layers.Dropout(0.1)(user)
            # item dense layer
            item = encoder.embedding(item_input)
            item = tf.keras.backend.squeeze(item, axis=1)
            item = tf.keras.layers.Dense(units=256, activation="relu")(item)
            item = tf.keras.layers.Dense(units=128, activation="relu")(item)
            item = tf.keras.layers.Dense(units=64, activation="relu")(item)
            # item = tf.keras.layers.Dropout(0.1)(item)
            # dot product
            logit = tf.keras.layers.Dot(axes=1)([user, item])
            pred = tf.keras.layers.Activation(activation='sigmoid')(logit)

        with tf.name_scope("train"):
            model = tf.keras.Model(inputs=[user_input, item_input], outputs=pred)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        return model

    def train(self):
        data, label = generate_train_dataset("./data/test.parquet")
        filepath = "./data/checkpoints/more_history/model-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     save_weights_only=True,
                                     verbose=1)
        self.model.fit(x = data,
                       y = label,
                       batch_size=self.batch_size,
                       shuffle=True,
                       epochs=self.epoch,
                       callbacks=[checkpoint])

    def test(self, test_file):

        with open(test_file) as fp:
            ids = [line[:-1] for line in fp]

        df = pd.read_parquet("./data/test.parquet")
        df = df.set_index("id")
        item = np.reshape(np.load("./data/test_1000.npy"), (-1, 1))
        with open("./data/dictionary.json") as fp:
            dictionary = json.load(fp)
        dictionary = dict([(value, key) for key, value in dictionary.items()])
        rec_fp = open("./recommend.txt", "w+")

        self.model.load_weights("./data/checkpoints/more_history/model-02.hdf5")

        for id in tqdm(ids):

            rec_fp.write("{} ".format(id))

            if not any(df.index.isin([id])):

                for i in range(100):
                    rec_fp.write("{} ".format(item[i][0]))
                rec_fp.write("\n")
                continue

            train = generate_evaluation_dataset(df, id, item.shape[0])
            pred = np.reshape(self.model.predict(x=[train, item], batch_size=self.batch_size), (-1))
            rec = pred.argsort()
            rec = rec[::-1]
            accuracy = [pred[idx] for idx in rec]
            name = [item[elem][0] for elem in rec]

            for elem in name:
                rec_fp.write("{} ".format(dictionary[elem]))
            rec_fp.write("\n")

            # name_dict = {}
            # for idx, key in enumerate(name):
            #     name_dict[key] = idx
            #
            # sample = df.loc[id]
            # for key in np.unique(sample["eval"]):
            #     if key in name_dict:
            #         print("\n", name_dict[key])
            #     else:
            #         print("\n KeyError")

        fp.close()

if __name__ == "__main__":
    model = Recommender()
#    model.train()
    model.test("./data/predict/dev.users")
#    Fire(Recommender)



