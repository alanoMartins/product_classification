import numpy as np
import tensorflow as tf

from preprocessor import Preprocessor
from model import bert_model, load_model, save_model


class Runner:

    def __init__(self, max_sequence, classes, model_path=None):
        self.max_sequence = max_sequence
        self.classes = classes

        self.preprocessor = Preprocessor(self.max_sequence)

        if model_path is None:
            self.model = bert_model(self.classes, self.max_sequence)
        else:
            self.model = load_model(model_path)

    def train(self, sequences, targets, epochs):
        labels = self.preprocessor.encoder_labels(targets)

        train, val = self.preprocessor.to_dataset(sequences, labels, self.max_sequence)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         patience=5,
                                                         restore_best_weights=True)

        self.model.fit(train, validation_data=val, epochs=epochs, callbacks=[tensorboard_callback, early_stopper])

        save_model(self.model, "../models/bert/bert_classify.h5")

    def prediction(self, data):
        Xids_sample, Xmask_sample = self.preprocessor.inputs_masks(data, self.max_sequence)
        predictions = self.model.predict([Xids_sample, Xmask_sample])
        return np.argmax(predictions, axis=1)


