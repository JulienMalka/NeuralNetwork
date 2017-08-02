import numpy as matrix


class NeuralNetwork:
    def __init__(self, learning_rate, layers):
        self.learning_rate = learning_rate
        self.layers = layers
        self.size = len(layers)
        self.weights = []
        self.bias = []
        for i in range(0, self.size-1):
            self.weights.append(matrix.random.randn(layers[i+1], layers[i]))
        for j in range(1, self.size):
            self.bias.append(matrix.random.randn(layers[j], 1))
        self.errors = []
        self.a_s = []
        self.z_s = []
        self.a = []
        self.z = []
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
        self.a.append(x)
        for i in range(0, self.size-1):
            self.z.append(matrix.dot(self.weights[i], self.a[i]))
            self.a.append(self.sigmoid(self.z[i]))
        #self.z1 = matrix.dot(self.w1, x) + self.b1
        #self.a1 = self.sigmoid(self.z1)
        #self.z2 = matrix.dot(self.w2, self.a1) + self.b2
        #self.a2 = self.sigmo0id(self.z2)
       # z = [self.z1, self.z2]
        self.z_s.append(self.z.copy())

        self.a_s.append(self.a.copy())
        return self.a[-1]

    def back_propagation(self, x, y):
        """Calculate the errors vectors for a given vector"""

        error = [0] * (self.size-1)
        y_hat = self.propagate(x)
        error[-1] = matrix.multiply(-(y - y_hat), self.sigmoid_prime(self.z[-1]))
        for i in range(1, self.size-1):
            error[-(i+1)] = matrix.dot(self.weights[-i].T, error[-i])*self.sigmoid_prime(self.z[-(i+1)])


        #error_1 = (matrix.dot(self.w2.T, error_2) * self.sigmoid_prime(self.z1))
        #error = [error_1, error_2]
        self.errors.append(error)
        self.a.clear()
        self.z.clear()


    def gradient_descent(self, m):
        """Updates the bias and weights using the errors vectors of all
        exemples that had been through backpropagation"""
        for a in range(0, self.size-1):
            tampon = self.errors[0][a]
            for i in range(1, len(self.errors)):
                tampon = tampon + self.errors[i][a]
            tampon = tampon * self.learning_rate / m
            self.bias[a] = self.bias[a] - tampon

        for b in range(0, self.size-1):
            tampon = matrix.dot(self.errors[0][b], matrix.transpose(self.a_s[0][b]))
            for i in range(1, len(self.errors)):
                tampon = tampon + matrix.dot(self.errors[i][b], matrix.transpose(self.a_s[i][b]))
            tampon = tampon * self.learning_rate / m
            self.weights[b] = self.weights[b] - tampon






        """tampon = self.errors[0][0]
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
        self.w2 = self.w2 - tampon"""

    def clear(self):
        """Clears the errors, as, and zs after a gradient descent"""
        self.errors.clear()
        self.a_s.clear()
        self.z_s.clear()
        self.a.clear()
        self.z.clear()
