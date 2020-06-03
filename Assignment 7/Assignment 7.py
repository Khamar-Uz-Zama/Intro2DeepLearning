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

maxTensor.indices

"""
Task 2
Given a tensor of shape (?, n), find the argmax in each row and return a new tensor 
that contains a 1 in each of the argmax’ positions, and 0s everywhere else.
"""


twoDTensor = tf.constant([ [1, 2, 3, 23],
                          [5, 40, 9, 1] ],tf.int16)

maxTensor = tf.math.top_k(twoDTensor, k=1, sorted=True)

tf.one_hot(maxTensor.indices, depth = twoDTensor.shape[1], dtype=tf.int32)


"""
Task 3
As in 1., but instead of “extracting” the top k values, create a new tensor with shape (?, n) 
where all but the top k values for each row are zero. 
Try doing this with a 1D tensor of shape (n,) (i.e. one row) first. 
Getting it right for a 2D tensor is more tricky; consider this a bonus.
Hint: You should look for a way to “scatter” a tensor of values into a different tensor. 
For two or more dimensions, you need to think carefully about the indices.
"""


"""
Task 4
Implement an exponential moving average. That is, given a decay rate a 
and an input tensor of length T,
create a new length T tensor where 
new[0] = input[0] and new[t] = a * new[t-1] + (1-a) * input[t] otherwise. 
Do not use tf.train.ExponentialMovingAverage.
"""
oneDTensor = tf.constant([1, 2, 3, 23], dtype = tf.float32)
decay = tf.constant(0.1)
new = []
for index,x in enumerate(oneDTensor):
    if (index == 0):
        new.append(x)
    else :
        s = tf.matmul(decay, new[index-1].numpy)
        a = tf.matmul((1-decay), x)
        new.append(s+a)
    
    
tf.map_fn(lambda index, value : value[0] if (index == 0) else decay * value[index-1] + (1-decay) * input[index],
          oneDTensor)
"""
Task 5
Find a way to return the last element in 4. without using loops. 
That is, return new[T] only – 
you don’t need to compute the other time steps (if you can avoid it).
"""







