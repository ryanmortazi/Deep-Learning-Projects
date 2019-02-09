# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:12:37 2018

@author: RyanM
"""

import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Defining the training/test data set
training_data_df = pd.read_csv("file path", dtype=float)
test_data_df = pd.read_csv("File Path", dtype=float)
X_training = 
Y_training = 
X_test = 
Y_test = 

#Data prepation : Scaling data
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)
X_scaled_test = X_scaler.transform(X_test)
Y_scaled_test = Y_scaler.transform(Y_test)

# Build a neural Network

# Define model parameters
# Choose your values for below param.
runName="run 1 with 50 nodes"
learning_rate = 0.001
training_epochs = 100

# Define how many inputs and outputs are in our neural network
number_of_inputs = 9
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Section One: Define the layers of the neural network itself

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights1", shape=[
        number_of_inputs, layer_1_nodes],
        initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[
        layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights)+biases)

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[
        layer_1_nodes, layer_2_nodes],
        initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[
        layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights)+biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[
        layer_2_nodes, layer_3_nodes],
        initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[
        layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights)+biases)

# Output Layer

with tf.variable_scope('Output'):
    weights = tf.get_variable(name="weights4", shape=[
        layer_3_nodes, number_of_outputs],
        initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[
        number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights)+biases

# Section Two: Define the cost function of the neural network that will
# measure prediction accuracy during training
# In this case we are using Mean Squared Error
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Create a summary operation to log the progress of the network
with tf.variable_scope('logging'):
    tf.summary.scalar("current_cost", cost)
    tf.summary.histogram("predicted_value", prediction)
    summary = tf.summary.merge_all()
    
# create a saver object
saver = tf.train.Saver()

# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:

    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    # Create log file writers to record training progress.
    # We'll store training and testing log data separately.
    training_writer = tf.summary.FileWriter("./logs/{}/training".format(runName), session.graph)
    testing_writer = tf.summary.FileWriter("./logs/{}/testing".format(runName), session.graph)

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={
                    X: X_scaled_training, Y: Y_scaled_training})
        # print cost every 5 epochs for training and test
        if epoch % 5 == 0:
            training_cost, training_summary = session.run([cost, summary], feed_dict={
                X: X_scaled_training, Y: Y_scaled_training})
            testing_cost, testing_summary = session.run(
                [cost, summary], feed_dict={X: X_scaled_test, Y: Y_scaled_test})

           # Write the current training status to the log files (Which we can view with TensorBoard)
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

           # Print the current training status to the screen
            print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch,
                                                                           training_cost, testing_cost))

    # print final cost for training and test
    final_training_cost = session.run(cost, feed_dict={
        X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(
        cost, feed_dict={X: X_scaled_test, Y: Y_scaled_test})
    print("final training cost : {}".format(final_training_cost))
    print("final testing cost : {}".format(final_testing_cost))

# save the model as trained_model and later on you can load the model and use it
    save_path = saver.save(session, "./logs/{}/trained_model".format(runName))
    print("model saved : {}".format(save_path))
    # Training is now complete!
    print("Training is complete!")
