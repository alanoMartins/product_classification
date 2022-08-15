import tensorflow as tf
from transformers import TFAutoModel


# class Bert_Classificaton:
#
#     def __init__(self):
#         self.bert = TFAutoModel.from_pretrained("bert-base-cased")
#
#     def build(self, num_classes, max_seq=128):
#         input_ids = tf.keras.layers.Input(shape=(max_seq,), name='input_ids', dtype='int32')
#         mask = tf.keras.layers.Input(shape=(max_seq,), name='attention_mask', dtype='int32')
#
#         # we consume the last_hidden_state tensor from bert (discarding pooled_outputs)
#         embeddings = self.bert(input_ids, attention_mask=mask)[0]
#
#         X = tf.keras.layers.LSTM(64)(embeddings)
#         X = tf.keras.layers.BatchNormalization()(X)
#         X = tf.keras.layers.Dense(64, activation='relu')(X)
#         X = tf.keras.layers.Dropout(0.1)(X)
#         y = tf.keras.layers.Dense(num_classes, activation='softmax', name='outputs')(X)
#
#         # define input and output layers of our model
#         model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
#
#         # freeze the BERT layer - otherwise we will be training 100M+ parameters...
#         model.layers[2].trainable = False
#
#         optimizer = tf.keras.optimizers.Adam(0.01)
#         loss = tf.keras.losses.CategoricalCrossentropy()  # categorical = one-hot
#         acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
#
#         model.compile(optimizer=optimizer, loss=loss, metrics=[acc])
#
#         return model


def bert_model(num_classes, max_seq=128):
    bert = TFAutoModel.from_pretrained("bert-base-cased")
    input_ids = tf.keras.layers.Input(shape=(max_seq,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(max_seq,), name='attention_mask', dtype='int32')

    # we consume the last_hidden_state tensor from bert (discarding pooled_outputs)
    embeddings = bert(input_ids, attention_mask=mask)[0]

    X = tf.keras.layers.LSTM(64)(embeddings)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(64, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    y = tf.keras.layers.Dense(num_classes, activation='softmax', name='outputs')(X)

    # define input and output layers of our model
    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

    # freeze the BERT layer - otherwise we will be training 100M+ parameters...
    model.layers[2].trainable = False

    optimizer = tf.keras.optimizers.Adam(0.01)
    loss = tf.keras.losses.CategoricalCrossentropy()  # categorical = one-hot
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    return model


def load_model(path):
    return tf.keras.models.load_model(path)


def save_model(model, path):
    model.save(path)