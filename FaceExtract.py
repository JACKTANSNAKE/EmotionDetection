import cv2
import glob
import os
from os import path

detPath = path.abspath("./haarcascades/haarcascade_frontalface_default.xml")
# print(detPath)
faceDet = cv2.CascadeClassifier(detPath)
detPath2 = path.abspath("./haarcascades/haarcascade_frontalface_alt2.xml")
faceDet_two = cv2.CascadeClassifier(detPath2)
detPath3 = path.abspath("./haarcascades/haarcascade_frontalface_alt0.xml")
faceDet_three = cv2.CascadeClassifier(detPath3)
detPath4 = path.abspath("./haarcascades/haarcascade_frontalface_alt_tree.xml")
faceDet_four = cv2.CascadeClassifier(detPath4)
emotions = ["Happy", "Sadness"]  # Define emotions
facesDBHappyPath = "./source_images/FacesDB/Happy"
facesDBSadPath = "./source_images/FacesDB/Sad"
facesDBHappy = []
facesDBSad = []

for file in os.listdir(facesDBHappyPath):
    facesDBHappy.append(os.path.join(facesDBHappyPath, file))
print(facesDBHappy)
for file in os.listdir(facesDBSadPath):
    facesDBSad.append(os.path.join(facesDBSadPath, file))
print(facesDBSad)


def detect_happy_faces(files):
    filenumber = 0
    for f in files:
        frame = cv2.imread(f)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        # Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
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
        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            print("face found in file: %s" % f)
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                cv2.imwrite("./source_images/dataset/Happy\\%s.jpg" % filenumber, out)  # Write image
            except:
                pass  # If error, pass file
        filenumber += 1  # Increment image number

def detect_sad_faces(files):
    filenumber = 0
    for f in files:
        frame = cv2.imread(f)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        # Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
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
        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            print("face found in file: %s" % f)
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                cv2.imwrite("./source_images/dataset/Sad\\%s.jpg" % filenumber, out)  # Write image
            except:
                pass  # If error, pass file
        filenumber += 1  # Increment image number


detect_happy_faces(facesDBHappy)
detect_sad_faces(facesDBSad)
