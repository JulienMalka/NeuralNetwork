import numpy as matrix
class NeuralNetwork:


    def __init__(self, nbInput, nbOutput, nbNeurons, learningRate):
        self.nbInput = nbInput
        self.nbOutput = nbOutput
        self.nbNeurons = nbNeurons
        self.W1 = matrix.random.randn(nbNeurons,nbInput)
        self.B1 = matrix.random.randn(nbNeurons,1)
        self.W2 = matrix.random.randn(nbOutput,nbNeurons)
        self.B2 = matrix.random.randn(nbOutput,1)
        self.errors = []
        self.As = []
        self.Zs = []
        self.learningRate = learningRate





    def sigmoid(self, z):
        return 1/(1+matrix.exp(-z))

    def sigmoidPrime(self, z):
        return matrix.exp(-z) / ((1 + matrix.exp(-z)) ** 2)


    def propagate(self, X):
        self.Z1 = matrix.dot(self.W1,X) + self.B1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = matrix.dot(self.W2, self.A1) + self.B2
        self.A2 = self.sigmoid(self.Z2)
        z = []
        z.append(self.Z1)
        z.append(self.Z2)
        self.Zs.append(z)
        a = []
        a.append(X)
        a.append(self.A1)
        self.As.append(a)
        return self.A2


    def backPropagation(self, X, Y):

        yHat = self.propagate(X)
        #print(Y-yHat)

        error_2 = matrix.multiply(-(Y - yHat),self.sigmoidPrime(self.Z2))
        error_1 = (matrix.dot(self.W2.T, error_2) * self.sigmoidPrime(self.Z1))
        error = [error_1,error_2]
        self.errors.append(error)




    def gradientDescent(self, M):

        tampon = self.errors[0][0]
        for i in range(0,len(self.errors)):
            tampon = tampon + self.errors[i][0]

        tampon = tampon * self.learningRate/M
        self.B1 = self.B1 - tampon

        tampon = self.errors[0][1]
        for i in range(0, len(self.errors)):
            tampon = tampon + self.errors[i][1]

        tampon = tampon * self.learningRate / M

        self.B2 = self.B2 - tampon
        tampon = matrix.dot(self.errors[0][0], matrix.transpose(self.As[0][0]))
        for i in range(0, len(self.errors)):
            tampon = tampon + matrix.dot(self.errors[i][0], matrix.transpose(self.As[i][0]))

        tampon = tampon * self.learningRate / M
        self.W1 = self.W1 - tampon

        tampon = matrix.dot(self.errors[0][1], matrix.transpose(self.As[0][1]))
        for i in range(0, len(self.errors)):
            tampon = tampon + matrix.dot(self.errors[i][1], matrix.transpose(self.As[i][1]))

        tampon = tampon * self.learningRate / M

        self.W2 = self.W2 - tampon


    def clear(self):
        self.errors.clear()
        self.As.clear()
        self.Zs.clear()





































