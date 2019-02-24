# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 19:01:52 2019

@author: 15146
"""
import tensorflow as tf
import numpy as np

def create_placeholders(n_x, n_y):
    """
   
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 28 * 28 * 1 channel = 784)
    n_y -- scalar, number of classes (from 0 to 9, so -> 10)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32,shape=(None,n_x),name="X")
    Y = tf.placeholder(tf.float32,shape=(None,n_y),name="Y")

    
    return X, Y

# =============================================================================
# ''' Testing the place holders function '''
# X, Y = create_placeholders(784, 10)
# print ("X = " + str(X))
# print ("Y = " + str(Y))
# 
# 
# =============================================================================
''' Initializing the parameters'''

def initialize_parameters(n_x, n_y,units,num__layers=3):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
    
    # Network parameters
    n_x -- scalar, size of an image vector (num_px * num_px = 28 * 28 * 1 channel = 784)
    n_y -- scalar, number of classes (from 0 to 9, so -> 10)
    n_hidden_1 = 128 # Units in first hidden layer
    n_hidden_2 = 128 # Units in second hidden layer
    
    So:
                        W1 : [784, 128]
                        b1 : [1, 128]
                        W2 : [128, 128]
                        b2 : [1, 128]
                        W3 : [128, 10]
                        b3 : [1, 10]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
    parameters={}
    parameters["W"+str(1)] = tf.get_variable("W"+str(1),[n_x,units[0]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    parameters["b"+str(1)] = tf.get_variable("b"+str(1), [1,units[0]], initializer = tf.zeros_initializer())
    
    for  i in range(2,num__layers+1):
        
        parameters["W"+str(i)] = tf.get_variable("W"+str(i),[units[i-2],units[i-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters["b"+str(i)] = tf.get_variable("b"+str(i), [1,units[i-1]], initializer = tf.zeros_initializer())
        
    return parameters

# =============================================================================
# 
# ''' testing the initializer '''
# 
# tf.reset_default_graph()
# with tf.Session() as sess:
#     parameters = initialize_parameters(n_x=784,n_y=10, units=[128,128,10])
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))
# =============================================================================
    
    
    
''' Forward propagation'''
    
def forward_propagation(X_placeholder, parameters, num__layers=3):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (number of examples,input size)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    temp_value_holder={}
    # Retrieve the parameters from the dictionary "parameters" 
        
    Z1 = tf.add(tf.matmul(X_placeholder,parameters["W"+str(1)]),parameters["b"+str(1)])              
    A1 = tf.nn.relu(Z1)       
    for i in range (2,num__layers):                  
        temp_value_holder["Z"+str(i)] = tf.add(tf.matmul(A1,parameters["W"+str(i)]),parameters["b"+str(i)])            
        temp_value_holder["A"+str(i)] = tf.nn.relu(temp_value_holder["Z"+str(i)])                         
   
    z_output= tf.add(tf.matmul(temp_value_holder["A"+str(num__layers-1)],parameters["W"+str(num__layers)]),parameters["b"+str(num__layers)])  
    temp_value_holder.clear()
    return z_output

# =============================================================================
# ''' testing forward prpagation '''
# tf.reset_default_graph()
# 
# with tf.Session() as sess:
#     X, Y = create_placeholders(784,10)
#     parameters = initialize_parameters(n_x=784,n_y=10, units=[128,128,10])
#     Z3 = forward_propagation(X, parameters)
#     print("Z3 = " + str(Z3))
#     
# ''' Forward propagation'''
# =============================================================================
   
def compute_cost(prediction, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples,10)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """  
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=Y))
    
    return cost

# =============================================================================
# ''' testing forward prpagation '''
# tf.reset_default_graph()
# 
# with tf.Session() as sess:
#     X, Y = create_placeholders(784,10)
#     parameters = initialize_parameters(n_x=784,n_y=10, units=[128,128,10])
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     print("cost = " + str(cost))
#     
# =============================================================================
