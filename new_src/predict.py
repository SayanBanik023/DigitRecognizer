import cv2
import numpy as np
# import os


class DigitRecognizer:
    def __init__(self, model):
        self.model = model
        # self.recent_predictions = [] # for smoothing results


    # @staticmethod
    # def preprocess_image(img, invert=True):
    #     """Convert image to grayscale, resize, normalize, and prepare for model."""
    #     if len(img.shape) == 3 and img.shape[2] == 3:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     if invert:
    #         img = cv2.bitwise_not(img)
    #     img = img / 255.0
    #     if img.shape != (28, 28):
    #         img = cv2.resize(img, (28, 28))
    #     return img.reshape(1, 28, 28, 1)
    #
    #
    # def predict_digit(self, img_path):
    #     img = cv2.imread(img_path)
    #     if img is None:
    #         raise ValueError(f"Could not read image at {img_path}")
    #
    #     processed_img = self.preprocess_image(img)
    #     prediction = self.model.predict(processed_img)
    #
    #     return {
    #         'digit' : int(np.argmax(prediction)),
    #         'confidence' : float(np.max(prediction)),
    #         'probabilities' : prediction[0].tolist()
    #     }
    #
    # def predict(self, image):
    #     imgs = np.array([image])
    #     res = self.model.predict(imgs)
    #     index = np.argmax(res)
    #     # print(index)
    #     return str(index)


    def predict(self, image):
        try:
            # Ensure the image is properly shaped for the model
            image = np.array(image, dtype=np.float32)  # Convert to float32
            image = image / 255.0  # Normalize pixel values (0-1 range)
            image = image.reshape(1, 28, 28, 1)  # Reshape for CNN input (batch size, height, width, channels)

            # Predict using the model
            predictions = self.model.predict(image)

            # Get the highest probability class
            predicted_digit = np.argmax(predictions)

            return str(predicted_digit)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None  # Return None if there's an error


