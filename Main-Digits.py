from NeuralNetwork import *
from mnist import MNIST
from Image import *
from random import shuffle
import numpy as np

NN = NeuralNetwork(28 * 28, 10, 30, 3)
data = MNIST('Data')
images_data, labels_data = data.load_training()
images_test, labels_test = data.load_testing()
batch_size = 10
epoch_number = 30
images = []


def to_vector(index):
    e = np.zeros((10, 1))
    e[index] = 1.0
    return e


def find_output(vector):
    return np.argmax(vector)

for o in range(0, len(images_data)):
    images.append(Image(images_data[o], labels_data[o]))

for j in range(0, epoch_number):
    h = 0
    shuffle(images)

    while h < len(images) - 10:

        for g in range(h, h + batch_size):
            image = np.array(images[g].image) / 255
            """Training phase"""
            NN.back_propagation(np.reshape(image, (28 * 28, 1)), to_vector(images[g].label))

        NN.gradient_descent(batch_size)
        NN.clear()
        h = h + batch_size

    count = 0
    for v in range(0, len(images_test)):

        """Testing phase"""
        image = np.array(images_test[v]) / 255

        if find_output(NN.propagate(np.reshape(image, (28 * 28, 1)))) == labels_test[v]:
            count = count + 1

    print("Epoch n°" + str(j + 1) + " --> Réussis : " + str(count) + "/10000 Making an accuracy of " + str(
        count * 100 / 10000) + "%")
