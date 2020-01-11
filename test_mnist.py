#This File's use to try the pre-trained Model from the Mnist model
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

#Load the pre-trained model
new_model = keras.models.load_model('mnist_classifier.model')

#to do the prediction, the model only accept list file or all the test data
predictions = new_model.predict([x_test])

#we can show the prediction of certain number by accessing the index of the data list
print(np.argmax(predictions[79]))

#we also can show the image of the prediction using pyplot
plt.imshow(x_test[79], cmap='binary')
plt.show()