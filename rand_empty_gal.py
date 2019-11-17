from opt import args
import galleries
import detectors
import vectorgenerator
import loaders
from PIL import Image
from cropper import crop_image

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import math
from tqdm import tqdm

gallery = galleries.options[args.gallery]()
detector = detectors.options[args.detector]()
vecgen = vectorgenerator.options[args.vectgen]()
dataloader = loaders.getLoader("../reid-data")

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

def general_display_results(results, save_name=None):
    # results = {"label":image}
    plot_size = int(math.ceil((len(results)/5)))
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(hspace=.7)
    row = plot_size
    columns = 5
    i = 1
    for label, image in results.items():
        fig.add_subplot(row, columns, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(image, interpolation="nearest")
        i += 1
    if save_name:
        plt.savefig(save_name+".png")
    plt.show()

index = 0
framecount = 0

moizpath = '../crops/moiz/fi004.jpg'
ethanpath = '../crops/ethan/f028.jpg'
sunpath = '../crops/sunjeon/f014.jpg'
ids = ['moiz', 'ethan', 'sun']
#gal = [getvec2out(moizpath, vecgen), getvec2out(ethanpath, vecgen), getvec2out(sunpath, vecgen)]
gal = list()
galimgs = list()

###############################################################
# Random set generation

marketpath = os.path.join('../../../Downloads', 'Market-1501-v15.09.15', 'bounding_box_train')
files = os.listdir(marketpath)
randimgs = list()
for i in range(0, 6):
    randimg = os.path.join(marketpath, files[i * 45])
    randimgs.append(getvec2out(randimg, vecgen))


###############################################################

outs = {
    "mov1": open("mov1.txt", "w"),
    "mov2": open("mov2.txt", "w"),
    "mov3": open("mov3.txt", "w")
}

successes = 0
totaltests = 0

for index, frames in tqdm(dataloader):
    for key, img in frames.items():
        boxes, scores = detector.getBboxes(img)
        idsinframe = list()
        for box in boxes:
            crop = (box[0][0].item(), box[0][1].item(), box[1][0].item(), box[1][1].item())
            img = img.crop(crop)
            uniqvect = vecgen.getVect2(img)
            comb = gal + randimgs
            dists = [np.average(np.dot(uniqvect, np.transpose(out2))) for out2 in comb]
            idents = getMaxIndex(dists, k=3)
            inds = idents[0]
            if inds > (len(gal) - 1):
                inds = len(gal)
                gal.append(uniqvect)
                galimgs.append(img)

            flagvalid = False
            for i in idents:
                if i not in idsinframe:
                    idsinframe.append(i)
                    inds = i
                    flagvalid = True
                    break
            if not flagvalid:
                inds = len(gal)
                gal.append(uniqvect)
                galimgs.append(img)
            # totaltests += 1
            # print(totaltests)
            # frame, x1, y1, x2, y2, id
            outs[key].write("%03d,%d,%d,%d,%d,%s\n" % (index, box[0][0], box[0][1], box[1][0], box[1][1], inds))
    # print(index)

labels = np.arange(0, len(galimgs))
labelsdict = {str(label): img for label, img in zip(labels, galimgs)}

general_display_results(labelsdict)

for file in outs.values():
    file.close()
