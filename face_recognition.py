import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model')
frontface_clsfr = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

source = cv2.VideoCapture(0)
with open('json/class_names.json', 'r') as file:
    labels = json.load(file)
with open('json/colors.json', 'r') as file:
    colors = json.load(file)
model.summary()

while True:
    _, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frontalfaces = frontface_clsfr.detectMultiScale(gray, 1.3, 3)

    for (x, y, w, h) in frontalfaces:
        face_img = gray[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (300, 300))
        resized = np.array(resized).astype(np.uint8)
        reshaped = tf.expand_dims(resized, axis=0)
        reshaped = tf.expand_dims(reshaped, axis=-1)
        result = model.predict(reshaped)
        label = int(np.argmax(result))
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[label % 7], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), colors[label % 7], -1)
        cv2.putText(img, labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('face recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
source.release()
