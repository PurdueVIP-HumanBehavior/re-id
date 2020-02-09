# TODO:(nhendy) script level docstring

import galleries
from detectors import FasterRCNN
from attribute_extractors import MgnWrapper
import loaders
from PIL import Image
from utils import crop_image
import argparse
import functools
import os
import sys
import numpy as np
import cv2
from sort import Sort
from bbox_trigger import BboxTrigger
from scipy.stats import mode
from tqdm import tqdm
from constants import *


def init_args():
    parser = argparse.ArgumentParser(description="multi-camera re-id system")
    parser.add_argument("-d",
                        "--detector",
                        help="Object detection model",
                        default='FasterRCNN',
                        choices=['FasterRCNN'])
    parser.add_argument("-r",
                        "--distance",
                        default='dot_product',
                        help="Distance metric used for retrieval",
                        choices=['dot_product'])
    parser.add_argument("-l",
                        "--loader",
                        default='video',
                        help="Type of data loading",
                        choices=['video'])
    parser.add_argument("-g",
                        "--gallery",
                        default='trigger',
                        help="Type of Gallery",
                        choices=['trigger'])
    parser.add_argument("-v",
                        "--vect_gen",
                        default='MGN',
                        help="Attribute extraction model",
                        choices=['MGN'])
    parser.add_argument("-i",
                        "--interval",
                        default=2,
                        help="Sampling interval",
                        type=int)
    parser.add_argument("--video_path",
                        required=True,
                        help="Path to the video to run the pipeline on")
    parser.add_argument(
        "--ref_image_path",
        required=True,
        help="Path to the reference image used for triggering",
    )
    parser.add_argument(
        "--weights_path",
        required=True,
        help="Path to MGN weigths",
    )
    return parser.parse_args()


def read_img_and_compute_feat_vector(path, attribute_extractor):
    if os.path.isfile(path):
        img = Image.open(path)
        return attribute_extractor(img)
    return None


def get_max_index(array, k=1):
    maxnums = array[0:k]
    maxinds = np.arange(0, k)
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
        return [x for _, x in sorted(zip(maxnums, maxinds), reverse=True)]


def load_predef_gallery_feat_vectors(imgs_dir, attribute_extractor):
    if not os.path.exists(path):
        raise ValueError("path doesn't exist")
    feat_vectors = list()
    for name in os.listdir(imgs_dir):
        feat_vector = read_img_and_compute_feat_vector(
            os.path.join(path, name), attribute_extractor)
        feat_vectors.append(feat_vector)
    return feat_vectors


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def main():
    args = init_args()

    detector = FasterRCNN()
    attribute_extractor = MgnWrapper(args.weights_path)
    dataloader = loaders.get_loader(args.video_path, args.loader,
                                    args.interval)

    ref_img = cv2.imread(args.ref_image_path)
    trigger_causes = [
        BboxTrigger(
            # TODO: (nhendy) weird hardcoded name
            "NE_Moiz",
            ref_img,
            DOOR_CLOSED_THRESHOLD,
            DOOR_OPEN_THRESHOLD,
            CHECK_OPEN_COORDS_TWO,
            TRIGGER_ROI_COORDS_TWO,
            detector,
        ),
        BboxTrigger(
            # TODO: (nhendy) weird hardcoded name
            "NE_Moiz",
            ref_img,
            DOOR_CLOSED_THRESHOLD,
            DOOR_OPEN_THRESHOLD,
            CHECK_OPEN_COORDS_ONE,
            TRIGGER_ROI_COORDS_ONE,
            detector,
        )
    ]

    gallery = galleries.TriggerGallery(attribute_extractor, trigger_causes)

    # create trackers for each video/camera
    trackers = {vidnames: Sort() for vidnames in dataloader.get_vid_names()}
    outfiles = {
        vidnames: open(vidnames + "tmp.txt", "w")
        for vidnames in dataloader.get_vid_names()
    }

    newfilenum = 0

    ###############################################################

    # iterate through frames of all cameras
    for findex, frames in tqdm(dataloader):

        # send frames from each camera to gallery to decide if references need to be captured based off triggering
        gallery.update(frames)

        # iterate through each camera
        for vidname, frame in frames.items():
            # get bounding boxes of all people
            boxes, scores = detector.get_bboxes(frame)

            # send people bounding boxes to tracker
            # get three things: normal Sort output (tracking bounding boxes it wants to send), corresponding track objects, and objects of new tracks
            tracker = trackers[vidname]
            dets = np.column_stack((np.reshape(boxes, [-1, 4]), scores))
            matched_tracks, matched_kb_trackers, new_kb_trackers = tracker.update(
                dets)

            # find indexes of returned bounding boxes that meet ideal ratio
            trkbboxes = np.array(matched_tracks)
            widths = trkbboxes[:, 2] - trkbboxes[:, 0]
            heights = trkbboxes[:, 3] - trkbboxes[:, 1]
            aspectratio = heights / widths
            readybools = np.isclose(aspectratio, 2, rtol=0.25)
            indexes = np.arange(len(matched_tracks))[readybools]

            # iterate through returned bounding boxes
            for ind, trk in enumerate(matched_tracks):
                box = ((int(trk[0]), int(trk[1])), (int(trk[2]), int(trk[3])))

                # if bounding box meets ideal ratio, save image of person as reference
                if ind in indexes:
                    cropimg = crop_image(frame, box)
                    if cropimg.size > 5:
                        newname = "tmpfiles/%07d.jpg" % newfilenum
                        newfilenum = newfilenum + 1
                        cv2.imwrite(newname, cropimg)
                        matched_kb_trackers[ind].save_img(newname)

                # write bounding box, frame number, and trackid to file
                outfiles[vidname].write("%d,%d,%.2f,%.2f,%.2f,%.2f\n" %
                                        (findex, trk[4], box[0][0], box[0][1],
                                         box[1][0], box[1][1]))

            # iterate through new tracks and add their current bounding box to list of track references
            for trk in new_kb_trackers:
                d = trk.get_state()[0]
                box = ((int(d[0]), int(d[1])), (int(d[2]), int(d[3])))
                cropimg = crop_image(frame, box)
                if cropimg.size > 5:
                    newname = "tmpfiles/%07d.jpg" % newfilenum
                    newfilenum = newfilenum + 1
                    cv2.imwrite(newname, cropimg)
                    trk.save_img(newname)

    # save images from gallery captured throughout video
    for i, img in enumerate(gallery.people):
        cv2.imwrite("tmpgal/%03d.jpg" % i, img)

    # load up the gallery (an artifact of debugging)
    gallery_feature_vectors = load_predef_gallery_feat_vectors(
        "tmpgal/", attribute_extractor)

    # iterate through trackers for each camera
    for vidname, sorto in trackers.items():
        tracks = sorto.trackers + sorto.rejects
        convertdict = dict()

        # iterate through tracks within each tracker
        for trk in tqdm(tracks):
            reidimgs = trk.imgfiles

            # iterate through every reference image for this track
            for img in reidimgs:
                # get feature vector of image
                uniqvect = read_img_and_compute_feat_vector(
                    img, attribute_extractor)

                if uniqvect is not None:
                    # find out what is the most similar gallery image
                    import ipdb
                    ipdb.set_trace()
                    dists = [
                        np.average(np.dot(uniqvect, np.transpose(out2)))
                        for out2 in gallery_feature_vectors
                    ]
                    index = get_max_index(dists, k=1)
                    trk.reid.append(index)

            # creating a dictionary mapping the trackIDs to the Re-IDs based on most frequent Re-ID of a track
            if len(trk.reid) > 0:
                import ipdb
                ipdb.set_trace()
                convertdict[trk.id] = mode(trk.reid)[0][0]

        # utility function to map track id to reID id
        def convert(val):
            if val in convertdict:
                return convertdict[val]
            else:
                return val

        # the fancy numpy thing
        convertnp = np.vectorize(convert)

        # use numpy thing to change trackIDs to Re-IDs in output file
        outfiles[vidname][:, 1] = convertnp(outfiles[vidname][:, 1])

        # save
        np.savetxt(vidname + ".txt", outfiles[vidname])


if __name__ == "__main__":
    sys.exit(main())
