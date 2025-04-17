
import keras
from keras._tf_keras.keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
import os


def create_model():
    model = keras.Sequential([
        Conv2D(filters = 32, kernel_size = (3,3), activation='relu', input_shape=(28,28,1)),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(filters = 64, kernel_size = (3,3), activation='relu',),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1) / 255.0

    model = create_model()

    model.fit(x_train, y_train, epochs = 4, validation_data=(x_test, y_test))

    os.makedirs('../models', exist_ok=True)
    model.save('../models/digit_recognizer_2.h5')


if __name__ == "__main__":
    main()