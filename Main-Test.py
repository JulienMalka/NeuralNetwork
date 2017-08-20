from NeuralNetwork import *
import numpy as np
NP = NeuralNetwork(0.1, [2, 3, 1])

X = np.array([[3], [5]])
X=(np.reshape(X, (2, 1)))
print(NP.propagate(X))
