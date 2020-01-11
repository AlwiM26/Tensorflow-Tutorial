#This file's is use to classify fashion_mnist dataset
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#import the fashion_mnist dataset from keras
fashion_mnist = tf.keras.datasets.fashion_mnist

#Split the dataset into train & test data for valuation
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Divide value of the train and test images pixel value to minimize calculation
train_images, test_images = train_images / 255.0, test_images / 255.0

#Make the classes name by the index of the labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Build the model, the model is Sequential or feed forward NN
model = tf.keras.models.Sequential([
    #The first layer is input layers from the image, and flatten the image into 1 dimension
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    #Than the the hidden layer with 128 neurons with relu activation function
    tf.keras.layers.Dense(128, activation='relu'),

    #Than the output layers with 10 neurons with softmax activation function
    tf.keras.layers.Dense(10, activation='softmax')
])

#Compile the model 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train the model
model.fit(train_images, train_labels, epochs=10)

#Evaluate the model to get the loss and the accuracy 
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy : ', test_accuracy)

#Do the prediction, the predict function in the TF take the input as list of data
predictions = model.predict(test_images)

#Save the model
model.save('Fashion.model')