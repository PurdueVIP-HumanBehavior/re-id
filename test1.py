from opt import args
import galleries
import detectors
import vectorgenerator
import loaders
from cropper import crop_image

import numpy
import cv2

gallery = galleries.options[args.gallery]()
detector = detectors.options[args.detector]()
# vecgen = vectorgenerator.options[args.vectgen]()
# vecgen.cuda()
# vecgen.eval()
dataloader = loaders.getLoader("../reid-data")


index = 0
framecount = 0

##############################################
# This program only produces cropped images from all three video streams
##############################################

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
            img.crop(crop).save(name, "JPEG")
            index = index + 1
            # if index == 500:
            #     quit(0)
    framecount = framecount + 1
