import cv2
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm

# TODO: (nhendy) this script needs massive clean up

frameind = 0
idind = 1
x1ind = 2
y1ind = 3
x2ind = 4
y2ind = 5
delimiter = ','

def main():
    args = init_args()
    create_vid(args.dest_vid, args.source_vid, args.ids_txt, view=args.view)


def init_args():
    parser = argparse.ArgumentParser(description="paints videos with bounding boxes and IDs")
    parser.add_argument("-v", "--view", default=False, action='store_true')
    parser.add_argument("source_vid", metavar='src', type=str,
                        help="the source video to use")
    parser.add_argument("ids_txt", metavar='ids', type=str,
                        help="the text file of format <frame, id, x1, y1, x2, y2>")
    parser.add_argument("dest_vid", metavar="dest", type=str,
                        help="the name of the destination video")
    return parser.parse_args()

def create_vid(savename, vid, outtxt, view=False):
    """
    creates a video using the out video
    :param savename: the name to save the video as
    :param vid: the video file path
    :param outtxt: the output text file
    :return: 1 for successful; 0 for failure
    """
    if not os.path.exists(vid):
        raise ValueError("vid " + vid + " does not exist")

    invid = cv2.VideoCapture(vid)
    if invid.isOpened() == False:
        raise RuntimeError("error opening file " + vid)

    if not os.path.exists(outtxt):
        raise ValueError("outtxt " + outtxt + "does not exist")

    output = np.loadtxt(outtxt, delimiter=delimiter)
    if output.shape[1] != 6:
        raise ValueError("The text file should have 6 entries per row. yours has {}".format(output.shape[1]))

    uniqframenums = np.sort(np.unique(output[:, frameind]))
    interval = np.average(uniqframenums[1:] - uniqframenums[:-1]).astype(np.int64)

    savename = "".join(savename.split('.')[:-1]) + ".avi"
    while os.path.exists(savename):
        decision = input("the video file " + savename + " already exists. Want to overwrite it? [y/n] ").lower()
        if decision == 'n':
            savename = input("new file name: ")
            savename = "".join(savename.split('.')[:-1]) + ".avi"
        else:
            break

    vidfps = int(invid.get(cv2.CAP_PROP_FPS) / interval)
    vidsize = (int(invid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(invid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    outvid = cv2.VideoWriter(savename + '.avi',
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    vidfps, vidsize)

    for frame in tqdm(uniqframenums):
        framerows = output[output[:, frameind]==frame, :]
        invid.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = invid.read()
        if ret:
            bboxes = framerows[:, x1ind:(y2ind+1)].astype(np.int64)
            ids = framerows[:, idind].astype(np.int64)
            nimg = paint_frame(img, bboxes, ids)
            if view:
                cv2.imshow("savename", nimg)
                cv2.waitKey(int(1000 / vidfps))
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
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    color=(0, 255, 0),
                    thickness=3)
    return img

if __name__ == "__main__":
    sys.exit(main())
