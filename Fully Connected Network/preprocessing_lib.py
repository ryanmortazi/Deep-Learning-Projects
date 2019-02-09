# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 18:34:58 2019

@author: 15146
"""

import tensorflow as tf
import math, numpy as np

'''
Example:
        
labels = np.array([1,2,3,0,2,1])
one_hot = preprocessing.one_hot_encoder(labels, C = 4)

output: 
    
one_hot = [[0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]
 [1. 0. 0. 0.]
 [0. 0. 1. 0.]
 [0. 1. 0. 0.]]

'''
def one_hot_encoder(labels, C):
       
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C,name="C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C,axis=0)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix).transpose()
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
       
    return one_hot



def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, input size)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape ( number of examples,1)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[0]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
  #  shuffled_Y = Y[:, permutation].reshape((1,m))
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[ k*mini_batch_size: (k+1)*mini_batch_size,:]
        mini_batch_Y = shuffled_Y[ k*mini_batch_size: (k+1)*mini_batch_size]
        ### END CODE HERE ###
        #mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches