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
sequenceLength = 200
data = data.map(lambda x: parse_seq(x, sequenceLength))

# a map from characters to indices
vocab = pickle.load(open("skp_vocab", mode="rb"))
vocab_size = len(vocab)
# inverse mapping: indices to characters
ind_to_ch = {ind: ch for (ch, ind) in vocab.items()}

#print(vocab)
#print(vocab_size)
units = 256
oneHotData = data.map(lambda char: tf.one_hot(indices = char, depth = vocab_size))

batch_OneHot_data = oneHotData.batch(batch_size=128, drop_remainder=True)

def initializer(hunits, vocab_size, sequenceLength):
    trainVariables = {}
    
    # Input to hidden layer
    trainVariables['W_InpToHidden'] = tf.Variable(tf.random_uniform_initializer()(shape=[hunits,vocab_size]))
    trainVariables['b_InpToHidden'] = tf.Variable(tf.random_uniform_initializer()(shape=[hunits,1]))
    
    # hidden to hidden layer
    trainVariables['W_HiddenToHidden'] = tf.Variable(tf.random_uniform_initializer()(shape=[hunits,hunits]))
    # Temp variable - remove later
    trainVariables['t-1_HiddenToHidden'] = tf.Variable(tf.random_uniform_initializer()(shape=[hunits,sequenceLength]))
    
    # hidden to output layer
    trainVariables['W_HiddenToOutput'] = tf.Variable(tf.random_uniform_initializer()(shape=[hunits,vocab_size]))
    trainVariables['b_HiddenToOutput'] = tf.Variable(tf.random_uniform_initializer()(shape=[vocab_size,1]))

    return trainVariables


trainVariables = initializer(units, vocab_size, sequenceLength)


for key,value in trainVariables.items():
    print(key, '' , value.shape)
    
    
n_time_steps = 100
train_batch = next(iter(batch_OneHot_data))
for time_step in range(n_time_steps):
    
    test_batch = next(iter(batch_OneHot_data))
    ## FeedForward computations
    for instance,label in zip(train_batch,test_batch):
        # First node
        op_InpToHidden = tf.matmul(trainVariables['W_InpToHidden'],tf.transpose(instance)) + trainVariables['b_InpToHidden']
        op_HiddenToHidden = tf.matmul(trainVariables['W_HiddenToHidden'],trainVariables['t-1_HiddenToHidden'])
        op_FromHidden = op_InpToHidden + op_HiddenToHidden
        op_FromHidden = tf.nn.tanh(op_FromHidden)
        
        # Second node
        op_FromOPLayer = tf.matmul(tf.transpose(trainVariables['W_HiddenToOutput']),op_FromHidden) + trainVariables['b_HiddenToOutput']
        y = tf.nn.softmax(logits = op_FromOPLayer)
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits = op_FromOPLayer))
    train_batch = test_batch