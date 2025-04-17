import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model
import os


class DigitRecognizer:
    def __init__(self, model_path='../models/digit_recognizer_1.h5'):
        self.model = load_model(model_path)

    @staticmethod
    def preprocess_image(img, invert=True):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if invert:
            img = cv2.bitwise_not(img)
        img = img / 255.0
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28))
        return img.reshape(1, 28, 28, 1)


    def predict_digit(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")

        processed_img = self.preprocess_image(img)
        prediction = self.model.predict(processed_img)

        return {
            'digit' : int(np.argmax(prediction)),
            'confidence' : float(np.max(prediction)),
            'probabilities' : prediction[0].tolist()
        }



def main():

    # global img_path
    recognizer = DigitRecognizer()
    i = 1

    while os.path.exists(f"../digits/{i}.png"):
        try:
            img_path = f"../digits/{i}.png"
            print(f"\nProcessing {img_path}...")

            prediction = recognizer.predict_digit(img_path)
            print(f"Predicted digit : {prediction['digit']}")
            print(f"Confidence : {prediction['confidence']:.2}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
        finally:
            i += 1


if __name__ == "__main__":
    main()



