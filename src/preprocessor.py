import numpy as np

import tensorflow as tf

from sklearn import preprocessing
from transformers import AutoTokenizer

tf.get_logger().setLevel('ERROR')


class Preprocessor:

    def __init__(self, max_sequence_size):
        self.tokenizer =  AutoTokenizer.from_pretrained("bert-base-cased")
        self.SEQ_LEN = max_sequence_size

    def encoder_labels(self, labels):
        le = preprocessing.LabelEncoder()
        arr = le.fit_transform(labels)
        labels = np.zeros((arr.size, arr.max()+1))
        labels[np.arange(arr.size), arr] = 1
        return labels

    def to_dataset(self, sequences, labels, max_seq):
        # initialize two arrays for input tensors
        Xids, Xmask = self.inputs_masks(sequences, max_seq)
        dataset = self.convert_dataset(Xids, Xmask, labels)
        return self.split_data(dataset)

    def tokenize(self, sentence):
        tokens = self.tokenizer.encode_plus(sentence, max_length=self.SEQ_LEN,
                                       truncation=True, padding='max_length',
                                       add_special_tokens=True, return_attention_mask=True,
                                       return_token_type_ids=False, return_tensors='tf')
        return tokens['input_ids'], tokens['attention_mask']

    def inputs_masks(self, sentences, max_seq):
        Xids = np.zeros((len(sentences), max_seq))
        Xmask = np.zeros((len(sentences), max_seq))

        for i, sentence in enumerate(sentences):
            Xids[i, :], Xmask[i, :] = self.tokenize(sentence)
            if i % 10000 == 0:
                print(i)
        return Xids, Xmask

    def convert_dataset(self, Xids, Xmask, labels):
        BATCH_SIZE = 32  # we will use batches of 32

        # load arrays into tensorflow dataset
        dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

        # create a mapping function that we use to restructure our dataset
        def map_func(input_ids, masks, labels):
            return {'input_ids': input_ids, 'attention_mask': masks}, labels

        # using map method to apply map_func to dataset
        dataset = dataset.map(map_func)

        # shuffle data and batch it
        return dataset.shuffle(10000).batch(BATCH_SIZE)

    def split_data(self, dataset):
        # get the length of the batched dataset
        DS_LEN = len([0 for batch in dataset])
        SPLIT = 0.9  # 90-10 split

        train = dataset.take(round(DS_LEN*SPLIT))  # get first 90% of batches
        val = dataset.skip(round(DS_LEN*SPLIT))  # skip first 90% and keep final 10%

        del dataset  # optionally, delete dataset to free up disk-space
        return train, val