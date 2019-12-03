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
import bboxtrigger
from scipy.stats import mode
from tqdm import tqdm

# gallery = galleries.options[args.gallery]()
detector = detectors.options[args.detector]()
vecgen = vectorgenerator.options[args.vectgen]()
# vecgen = vectorgenerator.options['TripleNet']()
dataloader = loaders.getLoader("../reid-data/msee2")

def getvec2out(path, vecgen):
    if os.path.isfile(path):
        moiz = Image.open(path)
        return vecgen.getVect2(moiz)
    return None

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
    retval = list()
    for name in imgfile:
        na = '.'.join(name.split('.')[:-1])
        img = getvec2out(os.path.join(path, name), vecgen)
        retval.append(img)
    return retval

###############################################################
def getVect(croppedimg):
    return vecgen.getVect2(croppedimg)

refimg = cv2.imread("../reid-data/msee2/NE_Moiz/00000.jpg")
chkcoord1 = [[1471, 67], [1487, 117]]
sampcoord1 = [[1348, 72], [1640, 671]]
trig1 = bboxtrigger.BboxTrigger('NE_Moiz', refimg, .27, .87, chkcoord1, sampcoord1, detector)
chkcoord2 = [[354, 70], [375, 110]]
sampcoord2 = [[114, 64], [600, 722]]
trig2 = bboxtrigger.BboxTrigger('NE_Moiz', refimg, .27, .84, chkcoord2, sampcoord2, detector)

gallery = galleries.TriggerGallery(getVect)
gallery.addTrigger(trig1)
gallery.addTrigger(trig2)

# cv2.namedWindow("vid", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("vid", 1067,600)

trackers = {vidnames: Sort() for vidnames in dataloader.getVidNames()}
outfiles = {vidnames: open(vidnames + 'tmp.txt', "w") for vidnames in dataloader.getVidNames()}

newfilenum = 0

mangall = loadPredefGal('../reid-data/gal1')

###############################################################

for findex, frames in tqdm(dataloader):

    gallery.update(frames)

    for vidname, frame in frames.items():
        boxes, scores = detector.getBboxes(frame)
        # dets = [box + [score] for box, score in zip(boxes, scores)]
        # boxes = np.reshape(boxes, [-1, 4])
        # print(boxes, boxes.shape)
        # print(scores, scores.shape)
        tracker = trackers[vidname]
        dets = np.column_stack((np.reshape(boxes, [-1, 4]), scores))
        tracks, trksoff, newtrks = tracker.update(dets)

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
                cropimg = crop_image(frame, box)
                if cropimg.size > 5:
                    newname = 'tmpfiles/%07d.jpg' % newfilenum
                    newfilenum = newfilenum + 1
                    cv2.imwrite(newname, cropimg)
                    trksoff[ind].save_img(newname)

            outfiles[vidname].write('%d,%d,%.2f,%.2f,%.2f,%.2f\n' % (findex, trk[4], box[0][0], box[0][1], box[1][0], box[1][1]))
                # cv2.rectangle(frame, box[0], box[1], color=(255, 0, 0), thickness=3)  # Draw Rectangle with the coordinates
            # else:
                # cv2.rectangle(frame, box[0], box[1], color=(0, 255, 0), thickness=3)
            # cv2.putText(frame, str(trk[4]), box[0], cv2.FONT_HERSHEY_SIMPLEX, 3, color=(0, 255, 0), thickness=3)
        for trk in newtrks:
            d = trk.get_state()[0]
            box = ((int(d[0]), int(d[1])), (int(d[2]), int(d[3])))
            cropimg = crop_image(frame, box)
            if cropimg.size > 5:
                newname = 'tmpfiles/%07d.jpg' % newfilenum
                newfilenum = newfilenum + 1
                cv2.imwrite(newname, cropimg)
                trk.save_img(newname)

        # cv2.imshow("vid", frame)
        # cv2.waitKey(100)
for i, img in enumerate(gallery.people):
    cv2.imwrite('tmpgal/%03d.jpg' % i, img)

for key in outfiles:
    outfiles[key].close()
    outfiles[key] = np.loadtxt(key + 'tmp.txt', delimiter=',')

    traks = trackers[key]
    tracks = traks.trackers + traks.rejects
    file = open(key + 'tmp2.txt', "w")
    for trk in tracks:
        file.write(str(trk.id) + ' ' + ' '.join(trk.imgfiles) + '\n')
    file.close()

for vidname, sorto in trackers.items():
    tracks = sorto.trackers + sorto.rejects
    convertdict = dict()
    for trk in tqdm(tracks):
        reidimgs = trk.imgfiles
        for img in reidimgs:
            uniqvect = getvec2out(img, vecgen)
            if uniqvect is not None:
                # dists = [np.average(np.dot(uniqvect, np.transpose(out2))) for out2 in gallery.feats]
                dists = [np.average(np.dot(uniqvect, np.transpose(out2))) for out2 in mangall]
                index = getMaxIndex(dists, k=1)
                trk.reid.append(index)

        if len(trk.reid) > 0:
            convertdict[trk.id] = mode(trk.reid)[0][0]

    def convert(val):
        if val in convertdict:
            return convertdict[val]
        else:
            return val

    convertnp = np.vectorize(convert)
    outfiles[vidname][:, 1] = convertnp(outfiles[vidname][:, 1])

    np.savetxt(vidname + '.txt', outfiles[vidname])



