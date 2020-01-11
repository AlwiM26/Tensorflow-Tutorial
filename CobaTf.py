import tensorflow as tf

mnist = tf.keras.datasets.mnist

#Prepare the dataset from keras using load_data method
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#convert the datasets into float datatype
x_train, x_test = x_train / 255.0, x_test / 255.0

#Create the sequential model, Flatten used to change the dataset input shape
#Dense used to change the data from Flatten layer into 1D Vector
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
