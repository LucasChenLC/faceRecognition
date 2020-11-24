import argparse
import cv2
from progressbar import *
import os


def collect_samples(sample_name, sample_size, mode):
    classifier = 'haarcascade_frontalface_alt2.xml'
    frontface_clsfr = cv2.CascadeClassifier(os.path.join('haarcascades', classifier))

    img_source = cv2.VideoCapture(0)
    widgets_ = ['Collecting: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                ' ', ETA(), ' ', FileTransferSpeed()]

    bar = ProgressBar(widgets=widgets_, maxval=100).start()
    data_path = os.path.join('dataset', sample_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        file = -1
    else:
        files = os.listdir(data_path)
        if not files:
            file = -1
        else:
            files.sort(key=lambda t: int(t[:-4]))
            file = files[-1]
            file = file.split('.')
            file = int(file[0])
    if mode == 'append':
        n = file + 1
    else:
        n = 0
    while True:
        _, img = img_source.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frontalfaces = frontface_clsfr.detectMultiScale(gray, 1.3, 3)
        if n >= sample_size:
            break
        for (x, y, w, h) in frontalfaces:
            face_img = gray[y:y + w, x:x + w]
            resized = cv2.resize(face_img, (300, 300))
            n += 1
            bar.update(n / sample_size * 100)
            cv2.imwrite(os.path.join(data_path, str(n)) + '.jpg', resized)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('capture', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    bar.finish()
    img_source.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default='Unknown', type=str, help='Input the name of the samples')
    parser.add_argument('-s', default=50, type=int, help='Input the size of the samples that you want to collect')
    parser.add_argument('-m', default='append', choices=['append', 'overwrite'], type=str,
                        help='Input the collect mode, append or overwrite')
    args = parser.parse_args()
    collect_samples(args.n, args.s, args.m)
