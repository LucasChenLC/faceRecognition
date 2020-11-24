import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('model')
a = 0
#for i in range(400, 501):
img = cv2.imread('/Users/bluesky/Desktop/faceRecognition/WJJ/401.jpg', cv2.IMREAD_GRAYSCALE)
src = np.array(img).astype(np.uint8)  # image: img (PIL Image):
src = tf.expand_dims(src, axis=0)
src = tf.expand_dims(src, axis=-1)
ans = model.predict(src)
print(ans)
labels = ['Lucas', 'WJJ']
x = labels[np.argmax(ans)]
if x != 'Lucas':
    a+=1
print(a)
