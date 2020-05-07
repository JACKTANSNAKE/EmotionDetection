import cv2
import numpy
from os import path

detPath = path.abspath("./haarcascades/haarcascade_frontalface_default.xml")
faceDet = cv2.CascadeClassifier(detPath)
detPath2 = path.abspath("./haarcascades/haarcascade_frontalface_alt2.xml")
faceDet_two = cv2.CascadeClassifier(detPath2)
detPath3 = path.abspath("./haarcascades/haarcascade_frontalface_alt0.xml")
faceDet_three = cv2.CascadeClassifier(detPath3)
detPath4 = path.abspath("./haarcascades/haarcascade_frontalface_alt_tree.xml")
faceDet_four = cv2.CascadeClassifier(detPath4)
font = cv2.FONT_HERSHEY_SIMPLEX
UpperLeftCornerOfText = (10, 30)
SecondUpperLeftCornerOfText = (100, 30)
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2
emotion = ["Happy", "Sad"]


def find_faces(image):
    face = faceDet.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
    # Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""
    # Cut and save face
    out = None
    x, y, w, h = 0, 0, 0, 0
    for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
        image = image[y:y + h, x:x + w]  # Cut the frame to size
        out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out = cv2.resize(out, (350, 350))  # Resize face so all images have same size
    return out, (x, y, w, h)


def showWebCamAndRun(model):
    """
    Shows webcam image, detects faces and its emotions in real time and draw emoticons over those faces.
    :param model: Learnt emotion detection model.
    :param window_size: Size of webcam image window.
    :param window_name: Name of webcam image window.
    """

    cam = cv2.VideoCapture(0)
    while True:

        ret, frame = cam.read()
        if frame is None:
            break

        f, (x, y, w, h) = find_faces(frame)
        if f is None:
            cv2.putText(frame, "Please put your face in front of the webcam!",
                                    UpperLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
            continue
        prediction = model.predict(f)
        confidence = 0
        if cv2.__version__ != '3.1.0':
            confidence = str(prediction[1])
            prediction = prediction[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion[prediction],
                    UpperLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(frame, confidence,
                    SecondUpperLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
    # cleanup the camera and close any open windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # load model
    p = path.abspath(f"./model/emotion_detection_model.xml")
    fisher_face = cv2.face.FisherFaceRecognizer_create()
    fisher_face.read(p)

    # use learnt model
    window_name = 'WEBCAM (press q to exit)'
    showWebCamAndRun(fisher_face)
