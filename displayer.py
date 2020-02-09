import cv2
import os
import sys
import numpy as np

# TODO: (nhendy) this script needs massive clean up

frameind = 0
idind = 1
x1ind = 2
y1ind = 3
x2ind = 4
y2ind = 5
delimiter = ','

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

def create_vid(savename, vid, outtxt, view=False):
    """
    creates a video using the out video
    :param savename: the name to save the video as
    :param vid: the video file path
    :param outtxt: the output text file
    :return: 1 for successful; 0 for failure
    """
    if not os.path.exists(vid):
        print("vid " + vid + " does not exist")
    invid = cv2.VideoCapture(savename)
    if invid.isOpened() == False:
        print("error opening file " + savename)

    if not os.path.exists(outtxt):
        print("outtxt " + outtxt + "does not exist")
    output = np.loadtxt(outtxt, delimiter=delimiter)
    uniqframenums = np.unique(output[frameind])
    interval = np.average(uniqframenums[1:] - uniqframenums[:-1]).astype(np.int64)

    while os.path.exists(savename):
        decision = input("the video file " + savename + " already exists. Want to overwrite it?[y/n]").lower()
        if decision == 'n':
            savename = input("new file name: ")
    vidfps = invid.get(cv2.CAP_PROP_FPS) / interval
    outvid = cv2.VideoWriter(savename + '.avi',
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    vidfps, (invid.get(cv2.CAP_PROP_FRAME_WIDTH), invid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    for frame in uniqframenums:
        framerows = output[output[:, frameind]==frame, :]
        invid.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = invid.read()
        if ret:
            bboxes = framerows[:, x1ind:(y2ind+1)]
            ids = framerows[:, idind]
            nimg = paint_frame(img, bboxes, ids)
            if view:
                cv2.imshow("savename", nimg)
                cv2.waitKey(1 / vidfps)
            outvid.write(nimg)
        else:
            break

    outvid.release()

def paint_frame(img, bboxes, ids):
    """
    paint an image with a list of bounding boxes and associated ids
    :param img: the frame as a numpy array
    :param bboxes: a list of bouding boxes - numpy array n x 4 array - each row is of form x1,y1,x2,y2
    :param ids: a list of ids for the corresponding bouding boxes
    :return: returns numpy array of image with bounding boxes painted on
    """
    for id, box in zip(ids, bboxes):
        cv2.rectangle(img,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      color=(0, 255, 0),
                      thickness=3)  # Draw Rectangle with the coordinates
        cv2.putText(img,
                    str(id),
                    box[0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    color=(0, 255, 0),
                    thickness=3)
    return img

if __name__ == "__main__":
    sys.exit(main())