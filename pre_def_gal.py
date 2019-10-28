from opt import args
import galleries
import detectors
import vectorgenerator
import loaders
from PIL import Image
from cropper import crop_image

import numpy
import cv2

gallery = galleries.options[args.gallery]()
detector = detectors.options[args.detector]()
vecgen = vectorgenerator.options[args.vectgen]()
dataloader = loaders.getLoader("../reid-data")

def getvec2out(path, vecgen):
    moiz = Image.open(path)
    return vecgen.getVect2(moiz)

index = 0
framecount = 0

moizpath = '../crops/moiz/fi004.jpg'
ethanpath = '../crops/ethan/f028.jpg'
sunpath = '../crops/sunjeon/f014.jpg'
ids = ['moiz', 'ethan', 'sun']
gal = [getvec2out(moizpath, vecgen), getvec2out(ethanpath, vecgen), getvec2out(sunpath, vecgen)]

outs = {
    "mov1": open("mov1.txt", "w"),
    "mov2": open("mov2.txt", "w"),
    "mov3": open("mov3.txt", "w")
}

for index, frames in dataloader:
    for key, img in frames.items():
        boxes, scores = detector.getBboxes(img)
        for box in boxes:
            # person = crop_image(img, box)
            # person_vec = vecgen(person)
            # id = gallery.getID(person_vec)
            name = '../crops/%s_%04d_%04d.jpg' % (key, framecount, index)
            w, h = img.size
            crop = (box[0][0].item(), box[0][1].item(), box[1][0].item(), box[1][1].item())
            img = img.crop(crop)
            uniqvect = vecgen.getVect2(img)
            dists = [numpy.average(numpy.dot(uniqvect, numpy.transpose(out2))) for out2 in gal]
            id = ids[dists.index(max(dists))]
            # frame, x1, y1, x2, y2, id
            outs[key].write("%03d,%d,%d,%d,%d,%s\n" % (index, box[0][0], box[0][1], box[1][0], box[1][1], id))
    print(index)
for file in outs.values():
    file.close()
