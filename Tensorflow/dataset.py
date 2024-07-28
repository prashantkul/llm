import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text


class Dataset():
    """ Class to load and preprocess the dataset """
    def __init__(self, dataset_name, batch_size, buffer_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dataset, self.info = tfds.load(name=dataset_name, with_info=True, as_supervised=True)
        self.train_dataset, self.val_dataset = self.dataset['train'], self.dataset['validation']
        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in self.train_dataset), target_vocab_size=2**13)
        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in self.train_dataset), target_vocab_size=2**13)
        self.input_vocab_size = self.tokenizer_pt.vocab_size + 2
        self.target_vocab_size = self.tokenizer_en.vocab_size + 2
        self.max_length = 20

    def encode(self, pt, en):
        pt = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        return pt, en

    def tf_encode(self, pt, en):
        result_pt, result_en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en

    def filter_max_length(self, x, y):
        return tf.logical_and(tf.size(x) <= self.max_length, tf.size(y) <= self.max_length)

    def make_batches(self):
        train_dataset = self.train_dataset.map(self.tf_encode)
        train_dataset = train_dataset.filter(self.filter_max_length)
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(self.buffer_size).padded_batch(self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = self.val_dataset.map(self.tf_encode)
        val_dataset = val_dataset.filter(self.filter_max_length).padded_batch(self.batch_size)
        return train_dataset, val_dataset

    def decode(self, en):
        return self.tokenizer_en.decode([i for i in en if i < self.tokenizer_en.vocab_size])
    