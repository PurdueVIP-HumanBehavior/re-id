from opt import args
from bboxtrigger import BboxTrigger
import os
import loaders
import cv2
import numpy as np


path = "../reid-data/msee2/NE_Moiz"
name = "00581.jpg"
firstimgfile = os.path.join(path, name)
# dataloader = loaders.getLoader("../reid-data/bidc")

# _, firstimg = next(iter(dataloader))
# firstimg = list(firstimg.values())[0]

def convertPILtoCV(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    # img = cv2.imread(firstimgfile)
    cv2.namedWindow("STrig", flags=cv2.WINDOW_NORMAL)
    othimg = cv2.imread(firstimgfile)
    bbox = cv2.selectROI("STrig", othimg)
    cv2.destroyWindow("STrig")

    cropimg = othimg[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
    cv2.imwrite(os.path.join(path, "crop" + name), cropimg)
    # for index, image in dataloader:

    print(bbox)

