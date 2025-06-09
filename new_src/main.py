import os
from prepare_model import get_mnist_data, train_model
# from predict import DigitRecognizer
from webcam import start_cv
from keras._tf_keras.keras.models import load_model

# def main():
#     # if a model is already saved just load it - else build it
#     model = None
#     try:
#         model = load_model('../models/digit_recognizer_1.h5')
#         print('loaded saved model.')
#         print(model.summary())
#     except:
#         # load and train data
#         print("getting mnist data...")
#         (x_train, y_train, x_test, y_test) = get_mnist_data()
#         print("training model...")
#         model = train_model(x_train, y_train, x_test, y_test)
#         print("saving model..."
#         os.makedirs('../models', exist_ok=True)
#         model.save('../models/digit_recognizer_1.h5')
#
#     global img_path
#     recognizer = DigitRecognizer(model)
#     i = 1
#
#     while os.path.exists(f"../digits/{i}.png"):
#         try:
#             img_path = f"../digits/{i}.png"
#             print(f"\nProcessing {img_path}...")
#
#             prediction = recognizer.predict_digit(img_path)
#             print(f"Predicted digit : {prediction['digit']}")
#             print(f"Confidence : {prediction['confidence']:.2}")
#
#         except Exception as e:
#             print(f"Error in processing {img_path}: {e}")
#         finally:
#             i += 1


def main():
    # if a model is already saved just load it - else build it
    model = None
    try:
        model = load_model('../models/digit_recognizer_4.h5')
        print('loaded saved model.')
        print(model.summary())
    except:
        # load and train data
        print("getting mnist data...")
        (x_train, y_train, x_test, y_test) = get_mnist_data()
        print("training model...")
        model = train_model(x_train, y_train, x_test, y_test)
        print("saving model...")
        os.makedirs('../models', exist_ok=True)
        model.save('../models/digit_recognizer_4.h5')
        print("date model is ready to use...")


    print("starting cv...")

    # show opencv window
    start_cv(model)

if __name__ == "__main__":
    main()