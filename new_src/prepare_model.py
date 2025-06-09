import keras
from keras._tf_keras.keras.layers import  (
    Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, LeakyReLU, Dropout, GlobalAveragePooling2D,
    DepthwiseConv2D, Input
)
import os

# get the mnist data
def get_mnist_data():
    # get mnist data
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


# create the model
def create_model():
    # model = keras.Sequential([
    #     Conv2D(filters = 32, kernel_size = (3,3), activation='relu', input_shape=(28,28,1)),
    #     BatchNormalization(),
    #     MaxPooling2D((2,2)),
    #
    #     Conv2D(filters = 64, kernel_size = (3,3), activation='relu',),
    #     BatchNormalization(),
    #     MaxPooling2D((2,2)),
    #
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     BatchNormalization(),
    #     Dense(10, activation='softmax')
    # ])

    # model = keras.Sequential([
    #     Flatten(input_shape=(28, 28)),
    #     Dense(512, activation='relu'),
    #     Dense(10, activation='softmax')
    # ])

    model = keras.Sequential([
        # First Conv Layer with larger kernel & He initialization
        Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Second Conv Layer with more filters and LeakyReLU
        Conv2D(64, (3, 3), kernel_initializer='he_uniform'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Third Conv Layer for deeper feature extraction
        Conv2D(128, (3, 3), kernel_initializer='he_uniform'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),

        # Replace Flatten with GlobalAveragePooling2D (reduces parameters)
        GlobalAveragePooling2D(),

        # Fully Connected Layers with Dropout for regularization
        Dense(64, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.3),

        # Output Layer with softmax activation
        Dense(10, activation='softmax')
    ])


    optimizer = keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # return model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(x_train, y_train, x_test, y_test):

    # callback
    # class myCallback(keras._tf_keras.keras.callbacks.Callback):
    #     def on_epoch_end(self, epoch, logs=None):
    #         if logs is None:
    #             logs = {}
    #         print(logs)
    #         if logs.get('accuracy') > 0.99:
    #             print("\nReached 99% accuracy so cancelling training!")
    #             self.model.stop_training = True

    import tensorflow as tf

    class MyCallback(keras._tf_keras.keras.callbacks.Callback):
        def __init__(self, target_accuracy=0.99, monitor='accuracy'):
            super(MyCallback, self).__init__()
            self.target_accuracy = target_accuracy
            self.monitor = monitor  # Can be 'accuracy' or 'val_accuracy'

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}

            acc = logs.get(self.monitor)
            print(f"Epoch {epoch + 1}: {self.monitor} = {acc:.4f}")  # Log the progress

            if acc and acc >= self.target_accuracy:
                print(f"\nReached {self.target_accuracy * 100}% {self.monitor}, stopping training early!")
                self.model.stop_training = True


    callbacks = MyCallback()

    # reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1) / 255.0

    # create model
    model = create_model()

    # fit model
    # model.fit(x_train, y_train, epochs = 4, validation_data=(x_test, y_test))
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    print(history.epoch, history.history['accuracy'][-1])

    return model
