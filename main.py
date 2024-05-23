import numpy as np
import cv2
import os
import tensorflow as tf
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

font = cv2.FONT_HERSHEY_PLAIN
font_scale = 0.8
blue = (255,0,0)
thickness = 1

classes = ['anger', 'fear', 'happy', 'neutral', 'sad']

model = tf.keras.models.load_model('final_model.keras')
model.summary()

def preprocess(img):
    if img.shape != (48, 48):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = cv2.resize(gray_img, (48, 48))
        tensor = tf.constant(processed, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, axis=0) / 255.0
        return tensor
    else:
        tensor = tf.constant(img, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, axis=0) / 255.0
        return tensor

def classify(img):
    logit = model.predict(img)
    prediction = tf.nn.softmax(logit)
    argmax = np.argmax(prediction, 1)
    return classes[argmax[0]]

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            im = frame[y:y + h, x:x + w]
            im = preprocess(im)
            predict = classify(im)
            frame = cv2.putText(frame, predict, (x, y), font, font_scale, blue, thickness, cv2.LINE_AA)

        cv2.imshow('Video', frame)

        c = cv2.waitKey(1)
        if c == 27:  # ESC character
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
