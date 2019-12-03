from opt import args
import galleries
import detectors
import vectorgenerator
import loaders
from PIL import Image
from cropper import crop_image

import os
import numpy as np
import cv2
from mySort import Sort

gallery = galleries.options[args.gallery]()
detector = detectors.options[args.detector]()
#vecgen = vectorgenerator.options[args.vectgen]()
vecgen = vectorgenerator.options['TripleNet']()
dataloader = loaders.getLoader("../reid-data/msee2")

def getvec2out(path, vecgen):
    moiz = Image.open(path)
    return vecgen.getVect2(moiz)

def getMaxIndex(array, k=1):
    maxnums = array[0:k]
    maxinds = np.arange(0,k)
    minins = min(maxnums)
    mininin = maxnums.index(minins)
    for i, val in enumerate(array):
        if val > minins:
            maxnums[mininin] = val
            maxinds[mininin] = i
            minins = min(maxnums)
            mininin = maxnums.index(minins)
    if len(maxnums) == 1:
        return maxinds[0]
    else:
        return [x for _,x in sorted(zip(maxnums, maxinds), reverse=True)]

def loadPredefGal(path):
    if not os.path.exists(path):
        raise ValueError("path doesn't exist")
    imgfile = os.listdir(path)
    retval = dict()
    for name in imgfile:
        na = '.'.join(name.split('.')[:-1])
        img = getvec2out(os.path.join(path, name), vecgen)
        retval[na] = img
    return retval


loadPredefGal('../reid-data/gal1')

###############################################################
cv2.namedWindow("vid", cv2.WINDOW_NORMAL)
cv2.resizeWindow("vid", 1067,600)

trackers = {vidnames: Sort() for vidnames in dataloader.getVidNames()}

path = '../reid-data/msee2'
vidname = 'NW_Serena'
files = os.listdir(os.path.join(path, vidname))

###############################################################

for frame in files:
    frame = cv2.imread(os.path.join(path, vidname, frame))
    boxes, scores = detector.getBboxes(frame)
    # dets = [box + [score] for box, score in zip(boxes, scores)]
    # boxes = np.reshape(boxes, [-1, 4])
    # print(boxes, boxes.shape)
    # print(scores, scores.shape)
    tracker = trackers[vidname]
    dets = np.column_stack((np.reshape(boxes, [-1, 4]), scores))
    tracks, _ = tracker.update(dets)

    # trkbboxes = np.array([trk.get_state() for trk in tracker.trackers])
    trkbboxes = np.array(tracks)
    widths = trkbboxes[:, 2] - trkbboxes[:, 0]
    heights = trkbboxes[:, 3] - trkbboxes[:, 1]
    aspectratio = heights / widths
    readybools = np.isclose(aspectratio, 2, rtol=.25)
    # indexes = np.arange(len(tracker.trackers))[readybools]
    indexes = np.arange(len(tracks))[readybools]


    for ind, trk in enumerate(tracks):
        box = ((int(trk[0]), int(trk[1])), (int(trk[2]), int(trk[3])))
        # print(type(box))
        if ind in indexes:
            cv2.rectangle(frame, box[0], box[1], color=(255, 0, 0), thickness=3)  # Draw Rectangle with the coordinates
        else:
            cv2.rectangle(frame, box[0], box[1], color=(0, 255, 0), thickness=3)
        cv2.putText(frame, str(trk[4]), box[0], cv2.FONT_HERSHEY_SIMPLEX, 3, color=(0, 255, 0), thickness=3)

    cv2.imshow("vid", frame)
    cv2.waitKey(100)
