# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:02:32 2020

@author: Khamar Uz Zama
"""

import tensorflow as tf			


"""
Task 1
Given a 2D tensor of shape (?, n), extract the k (k <= n) highest values for each row
into a tensor of shape (?, k). 
Hint: There might be a function to get the “top k” values of a tensor.
"""


twoDTensor = tf.constant([ [1, 2, 3],
                          [5, 4, 9] ],tf.int16)
maxTensor = tf.math.top_k(twoDTensor, k=1, sorted=True)

maxTensor.values

"""
Given a tensor of shape (?, n), find the argmax in each row and return a new tensor 
that contains a 1 in each of the argmax’ positions, and 0s everywhere else.
"""