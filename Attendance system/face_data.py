import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
i = 0
facedata = []
dataset_path = "./face_dataset/"

filename = input("Enter the name of new student : ")

while True:
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    k = 1

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    i += 1

    for face in faces[:1]:
        x, y, w, h = face

        offset = 5
        face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_selection = cv2.resize(face_offset, (100, 100))

        if i % 10 == 0:
            facedata.append(face_selection)
            print(len(facedata))

        cv2.imshow(str(k), face_selection)
        k += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("faces", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

facedata = np.array(facedata)
facedata = facedata.reshape((facedata.shape[0], -1))
print(facedata.shape)

np.save(dataset_path + filename, facedata)
print("Dataset saved at : {}".format(dataset_path + filename + '.npy'))
print(filename);

cap.release()
cv2.destroyAllWindows()
