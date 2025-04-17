from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.datasets import mnist


def evaluate_model(model_path = '../models/digit_recognizer_2.h5'):
    model = load_model(model_path)
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Loss : {loss:.4f}")


if __name__ == '__main__':
    evaluate_model()

