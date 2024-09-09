import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
[x_train, y_train], [x_test, y_test] = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs = 3)
#
# model.save('handwritten.h5')

model = tf.keras.models.load_model('handwritten.h5')

# loss, accuracy = model.evaluate(x_test, y_test)
#
# print(loss)
# print(accuracy)

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        # Read the image in grayscale
        img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)

        # Resize the image to 28x28
        img = cv2.resize(img, (28, 28))

        # Invert the image colors
        img = np.invert(img)

        # Reshape the image to match model input (1, 28, 28, 1)
        img = img.reshape(1, 28, 28, 1)

        # Make a prediction using the model
        prediction = model.predict(img)

        # Print the predicted digit
        print(f"This digit is a {np.argmax(prediction)}")

        # Display the image
        plt.imshow(img[0].reshape(28, 28), cmap=plt.cm.binary)
        plt.show()

    except Exception as e:
        # Print the error message
        print(f"Error: {e}")

    finally:
        # Move to the next image
        image_number += 1