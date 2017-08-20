from mnist import MNIST
from NeuralNetwork import *
import numpy as np

NN = NeuralNetwork(3, [28 * 28, 30, 10])
data = MNIST('Data')
images_data, labels_data = data.load_training()
images_test, labels_test = data.load_testing()
images_data = np.array(images_data)/255
images_test = np.array(images_test)/255


batch_size = 10
epoch_number = 30
images = []
NN.train(images_data, labels_data, 30, 10, images_test, labels_test)
