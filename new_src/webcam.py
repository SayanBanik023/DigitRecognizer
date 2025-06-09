import cv2
import numpy as np
from predict import DigitRecognizer  # your model wrapper

# Global variables
threshold = 150
startInference = False

# Toggle inference with left mouse click
def ifClicked(event, x, y, flags, param):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

# Trackbar callback
def on_threshold(x):
    global threshold
    threshold = x

# Resize and pad to 28x28, skip empty or tiny images
def resize_and_pad(img, size=28):
    h, w = img.shape
    if h == 0 or w == 0:
        return None

    scale = 20.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h == 0 or new_w == 0:
        return None

    img = cv2.resize(img, (new_w, new_h))
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img
    return padded

# Main OpenCV digit recognition function
def start_cv(model):
    global threshold

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Digits')
    cv2.setMouseCallback('Digits', ifClicked)
    cv2.createTrackbar('Threshold', 'Digits', threshold, 255, on_threshold)

    recognizer = DigitRecognizer(model)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        if startInference:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

            # Morphology to improve contour detection
            kernel = np.ones((3, 3), np.uint8)
            thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Left-to-right sort

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 5 or h < 5:
                    continue

                digit_img = thresh_img[y:y + h, x:x + w]
                digit_img = resize_and_pad(digit_img)
                if digit_img is None:
                    continue

                prediction = recognizer.predict(digit_img)
                if prediction is not None:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, str(prediction), (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Digits', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

