import cv2
import os
import numpy as np

# TODO: (nhendy) this script needs massive clean up


def main():
    name = "SW_Ethan"
    interval = 2
    path = os.path.join("../reid-data/msee2", name)
    imgs = os.listdir(path)
    imgs = [os.path.join(path, obj) for obj in imgs]

    frameind = 0
    idind = 1
    x1ind = 2
    y1ind = 3
    x2ind = 4
    y2ind = 5
    delimiter = ','

    with open(name + ".txt", "r") as fildat:
        rawdata = fildat.read()

    colors = {1: (255, 0, 0), "ethan": (0, 255, 0), 0: (0, 0, 255)}

    height, width, _ = cv2.imread(imgs[0]).shape
    out = cv2.VideoWriter(name + '.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          30 / interval, (width, height))

    rawdata = rawdata.split("\n")
    index = 0
    cv2.namedWindow("vid", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("vid", 1067, 600)

    while index < len(rawdata):
        line = rawdata[index]
        ref = line.split(delimiter)
        next = ref
        if ref[0] == '':
            break
        imgfile = imgs[int(ref[0]) - interval]
        # print(imgfile)
        image = cv2.imread(imgfile)
        while ref[0] == next[0]:
            # print(next[1], next[2], next[3], next[4])
            box = ((int(next[x1ind]), int(next[y1ind])), (int(next[x2ind]),
                                                          int(next[y2ind])))
            cv2.rectangle(image,
                          box[0],
                          box[1],
                          color=(0, 255, 0),
                          thickness=3)  # Draw Rectangle with the coordinates
            cv2.putText(image,
                        str(next[idind]),
                        box[0],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        color=(0, 255, 0),
                        thickness=3)
            index = index + 1
            next = rawdata[index].split(",")

        out.write(image)
        cv2.imshow("vid", image)
        cv2.waitKey(100)

    out.release()


if __name__ == "__main__":
    sys.exit(main())
