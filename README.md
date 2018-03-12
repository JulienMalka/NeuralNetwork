# Python Neural Network

This python project is an implementation of a neural network with a general number of hidden layers. One of the implementations provided in the examples is an handwritten digits classifier.

The file NeuralNetwork.py is a class describing a NeuralNetwork with 1 input layer, n hidden layers, 1 output layer.

It has a backPropagation and a gradientDescent method to train it.

The Main file is using the MNIST database to train a NeuronalNetwork with 30 neurons in the hidden layer, 28*28 neurons in the input layer (1 by pixel) and 10 neurons in the output layer with a stochastic gradient descent algorithm.

It achieve about 95% accuracy on the MNIST testing set

You have to drop the 4 MNIST files in a "Data" folder for the MNIST loader to work.
