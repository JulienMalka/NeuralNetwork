import numpy as matrix


class NeuralNetwork:
    def __init__(self, nb_input, nb_output, nb_neurons, learning_rate):
        self.nbInput = nb_input
        self.nbOutput = nb_output
        self.nbNeurons = nb_neurons
        self.w1 = matrix.random.randn(nb_neurons, nb_input)
        self.b1 = matrix.random.randn(nb_neurons, 1)
        self.w2 = matrix.random.randn(nb_output, nb_neurons)
        self.b2 = matrix.random.randn(nb_output, 1)
        self.errors = []
        self.a_s = []
        self.z_s = []
        self.learning_rate = learning_rate
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + matrix.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return matrix.exp(-z) / ((1 + matrix.exp(-z)) ** 2)

    def propagate(self, x):
        """Method used to get an input x go through the neuronal network and outputs a2"""
        self.z1 = matrix.dot(self.w1, x) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = matrix.dot(self.w2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)
        z = [self.z1, self.z2]
        self.z_s.append(z)
        a = [x, self.a1]
        self.a_s.append(a)
        return self.a2

    def back_propagation(self, x, y):
        """Calculate the errors vectors for a given vector"""
        y_hat = self.propagate(x)
        error_2 = matrix.multiply(-(y - y_hat), self.sigmoid_prime(self.z2))
        error_1 = (matrix.dot(self.w2.T, error_2) * self.sigmoid_prime(self.z1))
        error = [error_1, error_2]
        self.errors.append(error)

    def gradient_descent(self, m):
        """Updates the bias and weights using the errors vectors of all
        exemples that had been through backpropagation"""
        tampon = self.errors[0][0]
        for i in range(1, len(self.errors)):
            tampon = tampon + self.errors[i][0]

        tampon = tampon * self.learning_rate / m
        self.b1 = self.b1 - tampon

        tampon = self.errors[0][1]
        for i in range(1, len(self.errors)):
            tampon = tampon + self.errors[i][1]

        tampon = tampon * self.learning_rate / m

        self.b2 = self.b2 - tampon
        tampon = matrix.dot(self.errors[0][0], matrix.transpose(self.a_s[0][0]))
        for i in range(1, len(self.errors)):
            tampon = tampon + matrix.dot(self.errors[i][0], matrix.transpose(self.a_s[i][0]))

        tampon = tampon * self.learning_rate / m
        self.w1 = self.w1 - tampon

        tampon = matrix.dot(self.errors[0][1], matrix.transpose(self.a_s[0][1]))
        for i in range(1, len(self.errors)):
            tampon = tampon + matrix.dot(self.errors[i][1], matrix.transpose(self.a_s[i][1]))

        tampon = tampon * self.learning_rate / m
        self.w2 = self.w2 - tampon

    def clear(self):
        """Clears the errors, as, and zs after a gradient descent"""
        self.errors.clear()
        self.a_s.clear()
        self.z_s.clear()
