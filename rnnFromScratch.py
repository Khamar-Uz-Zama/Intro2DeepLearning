# -*- coding: utf-8 -*-
"""
Created on Thu May 21 02:36:39 2020

@author: Khamar Uz Zama
"""

from prepare_data import parse_seq
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.compat.v1 import enable_eager_execution
enable_eager_execution()

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

#print(vocab)
#print(vocab_size)

oneHotData = data.map(lambda z: tf.one_hot(indices = z, depth = vocab_size))

# Input to hidden layer
W_InpToHidden = tf.Variable(tf.random_uniform_initializer()(shape=[200,256]))
b_InpToHidden = tf.Variable(tf.random_uniform_initializer()(shape=[200,1]))

# hidden to hidden layer
W_HiddenToHidden = tf.Variable(tf.random_uniform_initializer()(shape=[200,256]))
# Temp
O_HiddenToHidden = tf.Variable(tf.random_uniform_initializer()(shape=[256,128]))

# hidden to output layer
W_HiddenToOutput = tf.Variable(tf.random_uniform_initializer()(shape=[128,68]))
b_HiddenToOutput = tf.Variable(tf.random_uniform_initializer()(shape=[68]))



# Random input just to confirm
batched_onehotencoded_data = oneHotData.batch(batch_size=128, drop_remainder=True)
tempData = batched_onehotencoded_data.batch(1)
#
a = tf.matmul(W_InpToHidden,tempData) + b_InpToHidden + tf.matmul(W_HiddenToHidden,O_HiddenToHidden)
opFromHidden = tf.nn.tanh(a)

opFromOPLayer=tf.matmul(opFromHidden,W_HiddenToOutput) + b_HiddenToOutput
y = tf.nn.softmax(tf.reshape(opFromOPLayer, shape = [68]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    