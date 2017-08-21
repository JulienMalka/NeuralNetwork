import numpy as matrix
from random import shuffle
from Exemple import *


class NeuralNetwork:
    def __init__(self, learning_rate, layers):
        self.learning_rate = learning_rate
        self.layers = layers
        self.size = len(layers)
        self.weights = []
        self.bias = []
        for i in range(0, self.size - 1):
            self.weights.append(matrix.random.randn(layers[i + 1], layers[i])/matrix.sqrt(layers[i+1]))
        for j in range(1, self.size):
            self.bias.append(matrix.random.randn(layers[j], 1))
        self.errors = []
        self.a_s = []
        self.z_s = []
        self.a = []
        self.z = []

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + matrix.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return matrix.exp(-z) / ((1 + matrix.exp(-z)) ** 2)

    @staticmethod
    def to_vector(index):
        e = matrix.zeros((10, 1))
        e[index] = 1.0
        return e

    @staticmethod
    def find_output(vector):
        return matrix.argmax(vector)

    def propagate(self, x):
        """Method used to get an input x go through the neuronal network and outputs a2"""
        self.clear()
        self.a.append(x)

        for i in range(0, self.size - 1):
            self.z.append(matrix.dot(self.weights[i], self.a[i]) + self.bias[i])
            self.a.append(self.sigmoid(self.z[i]))

        self.z_s.append(self.z.copy())
        retour = self.a.copy()
        del self.a[-1]

        self.a_s.append(self.a.copy())
        return retour[-1]

    def back_propagation(self, x, y):
        """Calculate the errors vectors for a given vector"""

        error = [0] * (self.size - 1)
        y_hat = self.propagate(x)
        error[-1] = matrix.multiply(-(y - y_hat), self.sigmoid_prime(self.z[-1]))
        for i in range(1, self.size - 1):
            error[-(i + 1)] = matrix.dot(self.weights[-i].T, error[-i]) * self.sigmoid_prime(self.z[-(i + 1)])

        self.errors.append(error)
        self.a.clear()
        self.z.clear()

    def gradient_descent(self, m):
        """Updates the bias and weights using the errors vectors of all
        exemples that had been through backpropagation"""
        for a in range(0, self.size - 1):
            tampon = self.errors[0][a]
            for i in range(1, len(self.errors)):
                tampon = tampon + self.errors[i][a]
            tampon = tampon * self.learning_rate / m
            self.bias[a] = self.bias[a] - tampon

        for b in range(0, self.size - 1):
            tampon = matrix.dot(self.errors[0][b], matrix.transpose(self.a_s[0][b]))
            for i in range(1, len(self.errors)):
                tampon = tampon + matrix.dot(self.errors[i][b], matrix.transpose(self.a_s[i][b]))
            tampon = tampon * self.learning_rate / m
            self.weights[b] = self.weights[b] - tampon

        self.clear()

    def clear(self):
        """Clears the errors, as, and zs after a gradient descent"""
        self.errors.clear()
        self.a_s.clear()
        self.z_s.clear()
        self.a.clear()
        self.z.clear()

    def train(self, exemples_input, exemples_output, epoch_number, batch_size, test_input, test_output):
        exemples = []
        for o in range(0, len(exemples_input)):
            exemples.append(Exemple(exemples_input[o], exemples_output[o]))

        for j in range(0, epoch_number):
            h = 0
            shuffle(exemples)

            while h < len(exemples) - batch_size:

                for g in range(h, h + batch_size):
                    image = matrix.array(exemples[g].ex_input)
                    self.back_propagation(matrix.reshape(image, (self.layers[0], 1)),
                                          self.to_vector(exemples[g].ex_output))

                self.gradient_descent(batch_size)
                self.clear()
                h = h + batch_size
            self.test(test_input, test_output, j)

    def test(self, test_input, test_output, batch_number):
        count = 0
        for v in range(0, len(test_input)):

            """Testing phase"""
            image = matrix.array(test_input[v])

            if self.find_output(self.propagate(matrix.reshape(image, (self.layers[0], 1)))) == test_output[v]:
                count = count + 1

        print(
            "Epoch n°" + str(batch_number + 1) + " --> Réussis : " + str(count) + "/10000 Making an accuracy of " + str(
                count * 100 / len(test_input)) + "%")
