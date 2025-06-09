import cv2, numpy as np
from predict import DigitRecognizer

# left mouse click handler
startInference = False


def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference


# threshold slider handler
threshold = 100


def on_threshold(x):
    global threshold
    threshold = x


# the opencv display loop
def start_cv(model):
    global threshold
    cap = cv2.VideoCapture(0)
    frame = cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', 150, 255, on_threshold)
    background = np.zeros((480, 640), np.uint8)
    frameCount = 0

    while True:
        ret, frame = cap.read()

        if (startInference):

            # frame counter for showing text
            frameCount += 1

            # black outer frame
            frame[0:480, 0:80] = 0
            frame[0:480, 560:640] = 0
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # apply threshold
            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)

            # get central image
            resizedFrame = thr[240 - 75:240 + 75, 320 - 75:320 + 75]
            background[240 - 75:240 + 75, 320 - 75:320 + 75] = resizedFrame

            # resize for inference
            iconImg = cv2.resize(resizedFrame, (28, 28))

            # creating object of DigitRecognizer
            recognizer = DigitRecognizer(model)

            # pass to model predictor
            res = recognizer.predict(iconImg)

            # clear background
            if frameCount == 5:
                background[0:480, 0:80] = 0
                frameCount = 0

            # show text
            cv2.putText(background, res, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(background, (320 - 80, 240 - 80), (320 + 80, 240 + 80), (255, 255, 255), thickness=3)

            # display frame
            cv2.imshow('background', background)
        else:
            # display normal video
            cv2.imshow('background', frame)

        # cv2.imshow('resized', resizedFrame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
