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

index = 0
framecount = 0

moizpath = '../crops/moiz/fi004.jpg'
ethanpath = '../crops/ethan/f028.jpg'
sunpath = '../crops/sunjeon/f014.jpg'
ids = ['moiz', 'ethan', 'sun']
gal = [getvec2out(moizpath, vecgen), getvec2out(ethanpath, vecgen), getvec2out(sunpath, vecgen)]
galimgs = [moizpath, ethanpath, sunpath]
#gal = list()

###############################################################
# Random set generation

marketpath = os.path.join('../../../Downloads', 'Market-1501-v15.09.15', 'bounding_box_train')
files = os.listdir(marketpath)
randimgs = list()
for i in range(0, 6):
    randimg = os.path.join(marketpath, files[i * 45])
    randimgs.append(getvec2out(randimg, vecgen))


###############################################################

moizs = ['../crops/moiz/' + img for img in os.listdir('../crops/moiz')]
ethans = ['../crops/ethan/' + img for img in os.listdir('../crops/ethan')]
suns = ['../crops/sunjeon/' + img for img in os.listdir('../crops/sunjeon')]
lowestnum = min([len(moizs), len(ethans), len(suns)])
moizs = moizs[0:lowestnum]
ethans = ethans[0:lowestnum]
suns = suns[0:lowestnum]

tests = {
    'moiz': moizs,
    'ethan': ethans,
    'sun': suns
}

successes = 0
totaltests = 0

for key, imgs in tests.items():
    for img in imgs:
        out = getvec2out(img, vecgen)
        comb = gal + randimgs
        dists = [np.average(np.dot(out, np.transpose(out2))) for out2 in comb]
        inds = getMaxIndex(dists, k=1)
        if inds > (len(gal) - 1):
            gal.append(out)
            galimgs.append(img)
        else:
            if inds < len(ids):
                if ids[inds] == key:
                    successes += 1
        totaltests += 1
        print(totaltests)

print(successes/totaltests, len(gal))
print(galimgs)

