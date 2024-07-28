import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text

from typing import Iterator, Optional, Tuple


class Dataset():
    """ Class to load and preprocess the dataset """
    def __init__(self):
        self.max_length = 20
        self.MAX_TOKENS = 128
        self.BUFFER_SIZE = 20000
        self.BATCH_SIZE = 64
        self.tokenizers = self.build_tokenizer()

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

    def decode(self, en):
        return self.tokenizer_en.decode([i for i in en if i < self.tokenizer_en.vocab_size])
    
    def build_tokenizer(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = 'ted_hrlr_translate_pt_en_converter'
            
        tf.keras.utils.get_file(
                                f'{model_name}.zip',
                                f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
                                cache_dir='.', cache_subdir='', extract=True
                            )

        tokenizers = tf.saved_model.load(model_name)
        return tokenizers
    
    def tokenize(self, en_examples, tokenizers):
        return tokenizers.en.tokenize(en_examples)
        
    
    def detokenize(self, encoded, tokenizers):
        return tokenizers.en.detokenize(encoded)

    def tokens(self, encoded, tokenizers):
        return tokenizers.en.lookup(encoded)
    
    def prepare_batch(self, pt, en):
        pt = self.tokenizers.pt.tokenize(pt)
        pt = pt[:, :self.MAX_TOKENS]
        pt = pt.to_tensor()
        
        en = self.tokenizers.en.tokenize(en)
        en = en[:, :(self.MAX_TOKENS+1)]
        en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
        en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens
        
        return (pt, en_inputs), en_labels
    
    def make_batches(self, ds):
            return (
                ds
                .shuffle(self.BUFFER_SIZE)
                .batch(self.BATCH_SIZE)
                .map(self.prepare_batch, tf.data.AUTOTUNE)
                .prefetch(buffer_size=tf.data.AUTOTUNE))