import cv2
import os

path = '../reid-data/msee_atrium'
name = 'msee_NEside_Moiz'
ext = '.mp4'
cap = cv2.VideoCapture(os.path.join(path, name + ext))
if name not in os.listdir(path):
    os.makedirs(os.path.join(path, name))

if cap.isOpened() == False:
    print("error openning file")

i = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        namenum = '%05d'%i
        cv2.imwrite(os.path.join(path, name, namenum + '.jpg'), frame)
        if i % 100 == 0:
            print(i)
        i = i + 1
    else:
        break

cap.release()