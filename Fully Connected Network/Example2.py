# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:41:15 2019

@author: 15146
"""

import tensorflow as tf
from tensorflow import keras
# Helper libraries
from tensorflow.python.framework import ops
import sys
#sys.path.insert(0, '/path/to/application/app/folder')
sys.path.insert(0, 'C:/Users/15146/Desktop/CovNet_Tutorial/deep-Learning-projects/Fully Connected Network')
import preprocessing_lib 
import deep_learn_lib as dp

fashion_mnist = keras.datasets.fashion_mnist

'''Flatten the training and test images'''
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=train_images.shape))
print("Training set (labels) shape: {shape}".format(shape=train_labels.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=test_images.shape))
print("Test set (labels) shape: {shape}".format(shape=test_labels.shape))

X_train_flatten = train_images.reshape(-1, 28*28)
X_test_flatten = test_images.reshape(-1,28*28)

''' Normalize image vectors'''
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

'''# Convert training and test labels to one hot matrices'''

Y_train = preprocessing_lib.one_hot_encoder(train_labels, 10)
Y_test = preprocessing_lib.one_hot_encoder(test_labels, 10)

def logger(prediction,cost):
   # Create a summary operation to log the progress of the network
    with tf.variable_scope('logging'):
        tf.summary.scalar("current_cost", cost)
        tf.summary.histogram("predicted_value", prediction)
        summary = tf.summary.merge_all()
    return summary

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape ( number of training examples = 60000, input size = 784)
    Y_train -- training set, of shape (number of training examples = 60000, output size = 10)
    X_test -- test set, of shape (number of test examples = 10000, input size = 784)
    Y_test -- test set, of shape (umber of test examples = 10000, output size = 10)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (m,n_x) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[1]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = dp.create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = dp.initialize_parameters(n_x=784,n_y=10, units=[128,128,10],num__layers=3)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = dp.forward_propagation(X, parameters,num__layers=3)
    
    # Cost function: Add cost function to tensorflow graph
    cost = dp.compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Create a summary operation to log the progress of the network
    summary=logger(prediction=Z3,cost=cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    # create a saver object
    saver = tf.train.Saver()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        # Create log file writers to record training progress.
        # We'll store training and testing log data separately.
        training_writer = tf.summary.FileWriter("./logs/training", sess.graph)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = preprocessing_lib.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", 
                # the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                training_summary=sess.run(summary,feed_dict={X:X_train,Y:Y_train})
                training_writer.add_summary(training_summary, epoch)
                

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3,axis=1), tf.argmax(Y,axis=1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        # save the model as trained_model 
        save_path = saver.save(sess, "./logs/trained_model.cpkt")
        print("model saved : {}".format(save_path))
        # Training is now complete!
        print("Training is complete!")
        

############### Train the model#####################
        
model(X_train, Y_train, X_test, Y_test,num_epochs = 150)

'''
to visualize the cost graph in tensorboard, we need to 
1) run the below command line:
    
<< tensorboard --logdir="./logs/training" >>

2) Go to << localhost:6006>> 
'''

############# Load the trained model######################

# to be able to rerun the model without overwriting tf variables
ops.reset_default_graph()
# Create Placeholders of shape (n_x, n_y)
X, _ = dp.create_placeholders(n_x=784, n_y=10)
# Initialize parameters
parameters = dp.initialize_parameters(n_x=784,n_y=10, units=[128,128,10],num__layers=3)
# Forward propagation: Build the forward propagation in the tensorflow graph
Z = dp.forward_propagation(X, parameters,num__layers=3)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "./logs/trained_model.cpkt")
    parameters=sess.run(parameters)
    predictions=sess.run(Z,feed_dict={X:X_test})
    correct_prediction = tf.equal(tf.argmax(predictions,axis=1), tf.argmax(Y_test,axis=1))
    accuracy = sess.run(tf.reduce_mean(tf.cast(correct_prediction, "float")))
    print ("Test Accuracy:", accuracy)