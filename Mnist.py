import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

#Import all the Mnist dataset from keras library
mnist = keras.datasets.mnist

#Divide all the data into train and testing data
#Where x is the data and y is the label
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#change the into float number to ease the training process
x_train, x_test = x_train / 255.0, x_test / 255.0

#build the model
model = keras.models.Sequential([
    #first flatten / change the data dimension into 28x28 2d array
    keras.layers.Flatten(input_shape=(28, 28)),
    #than the next layer is 128 neuron using relu as the activation function
    keras.layers.Dense(128, activation='relu'),
    #than the dropout/learning rate is 0.2
    keras.layers.Dropout(0.2),
    #than for the output is gonna be 10 class of output with softmax activation function
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train the model using fit function with 5 epoch
model.fit(x_train, y_train, epochs=5)

#evaluate the model using the testing data
model.evaluate(x_test, y_test, verbose=2)

model.save('mnist_classifier.model')


#To see the image of the data, first reshape the data to 28x28 2d array
#and display it using pyplot
# plt.imshow(pixels, cmap='binary')
# plt.show()