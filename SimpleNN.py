import tensorflow as tf
from tensorflow import keras
import numpy as np

#use 1 layer of NN with 1 input neuron
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
#sgd is the optimizer used to optimize the nn with a new fake input
#the mean squared error is the loss function tu calculate the difference between
#the output and the real label
model.compile(optimizer='sgd', loss='mean_squared_error')

#the formula is y = 2x + 1 
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0 ], dtype=float)
ys = np.array([-1.0, 1.0, 3.0, 5.0, 7.0, 9.0 ], dtype=float)

#to train the nn using the xs as the input and ys as the labels
#epoch is the iteration number for the training
model.fit(xs, ys, epochs=500)
print(model.predict([20.0]))