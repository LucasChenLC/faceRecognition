from PIL import Image
import cv2
import numpy as np
import os

ori_path = 'dataset'
classifier = 'haarcascade_frontalface_alt2.xml'
face_clsfr = cv2.CascadeClassifier(os.path.join('haarcascades/', classifier))

print('Cropping the Image under classifier: ', classifier)

for dir in os.listdir(ori_path):
    for file in os.listdir(os.path.join(ori_path, dir)):
        img = cv2.imread(os.path.join(ori_path, dir, file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(gray, 1.3, 5)
        if len(faces) != 0:
            print(file, len(faces), 'faces detected')
        else:
            print(file, 'no faces detected')
            os.remove(os.path.join(ori_path, dir, file))
        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (300, 300))
            cv2.imwrite(os.path.join(ori_path, dir, file), face_img)