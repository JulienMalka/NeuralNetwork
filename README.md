# Python Neural Network

## Introduction
This python project is an implementation of a neural network with a general number of hidden layers. One of the implementations provided in the examples is an handwritten digits classifier.

The file NeuralNetwork.py is a class describing a NeuralNetwork with 1 input layer, n hidden layers, 1 output layer.

It has a backPropagation and a gradientDescent method to train it.

The Main file is using the MNIST database to train a NeuronalNetwork with 30 neurons in the hidden layer, 28*28 neurons in the input layer (1 by pixel) and 10 neurons in the output layer with a stochastic gradient descent algorithm.

It achieve about 95% accuracy on the MNIST testing set

You have to drop the 4 MNIST files in a "Data" folder for the MNIST loader to work.


## Documentation : The NeuralNetwork class

You can create a new *Neural Network* using the constructor, it takes a learning rate and a vector corresponding to the number of neuron in each layer. Example : ``NeuralNetwork(0.1, [3, 5, 8])`` would create a neural network with 3 layers of size 3, 5 and 8 respectively and with learning rate 0.1.


The NeuralNetwork class includes a few helper functions to work on the network such as ``propagate``, ``back_propagation``, ``gradient_descent``, but you can simply use the ``train`` function that does all that for you. It takes an exemple input and output, a number of epoch and an epoch size, a test input and output and does all the training and printing of result for you.


## Collaboration

Any collaboration is welcome !
