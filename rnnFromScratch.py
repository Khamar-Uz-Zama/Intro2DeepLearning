# -*- coding: utf-8 -*-
"""
Created on Thu May 21 02:36:39 2020

@author: user
"""

from prepare_data import parse_seq
import pickle
import tensorflow as tf

# this is just a datasets of "bytes" (not understandable)
data = tf.data.TFRecordDataset("skp.tfrecords")

# this maps a parser function that properly interprets the bytes over the dataset
# (with fixed sequence length 200)
# if you change the sequence length in preprocessing you also need to change it here
data = data.map(lambda x: parse_seq(x, 200))

# a map from characters to indices
vocab = pickle.load(open("skp_vocab", mode="rb"))
vocab_size = len(vocab)
# inverse mapping: indices to characters
ind_to_ch = {ind: ch for (ch, ind) in vocab.items()}

print(vocab)
print(vocab_size)