import cv2

name = 'mov3'
cap = cv2.VideoCapture(name + '.MOV')

if cap.isOpened() == False:
    print("error openning file")

i = 0
prefix = name + '/'

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        namenum = '%05d'%i
        cv2.imwrite(prefix + namenum + '.jpg', frame)
        if i % 100 == 0:
            print(i)
        i = i + 1
    else:
        break

cap.release()