import os
from datetime import datetime as dt

import cv2
import numpy as np
from flask import Flask, render_template, Response
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nl = []
        for line in myDataList:
            entry = line.split(',')
            nl.append(entry[0])
        if name not in nl:
            now = dt.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
def knn(tr, test, k=5):
    dist = []
    for i in range(tr.shape[0]):
        xi = tr[i, :-1]
        yi = tr[i, -1]
        d = distancee(test, xi)
        dist.append([d, yi])
    kd = sorted(dist, key=lambda x: x[0])[:k]
    label = np.array(kd)[:, -1]

    out_put = np.unique(label, return_counts=True)
    index = np.argmax(out_put[1])
    return out_put[0][index]

def distancee(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
dataset_path = "./face_dataset/"
face_data = []
label = []
class_id = 0
namess = {}
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        namess[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(label, axis=0).reshape((-1, 1))

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX
app = Flask(__name__)
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for face in faces:
                x, y, w, h = face
                offset = 5
                face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
                face_section = cv2.resize(face_section, (100, 100))

                out = knn(trainset, face_section.flatten())

                cv2.putText(frame, namess[int(out)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                markAttendance(namess[int(out)]);
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=5059,debug=True)