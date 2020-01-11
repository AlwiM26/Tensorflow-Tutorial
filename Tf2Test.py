#This file's use to try the trained fashion mnist model
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

dataset = tf.keras.datasets.fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

model = tf.keras.models.load_model('Fashion.model')

predictions = model.predict(train_images)
print(class_names[np.argmax(predictions[19])])
plt.imshow(test_images[19], cmap='binary')
plt.show()