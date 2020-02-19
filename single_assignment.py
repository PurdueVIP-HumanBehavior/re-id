# TODO:(nhendy) script level docstring
import tempfile
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
from third_party.sort import Sort
from bbox_trigger import BboxTrigger
from scipy.stats import mode
from scipy.optimize import linear_sum_assignment
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
    parser.add_argument(
        "--gallery_path",
        default="tmpgal/",
        help="Path to gallery",
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


def load_gallery_feat_vectors(imgs_dir, attribute_extractor):
    if not os.path.exists(imgs_dir):
        raise ValueError("path doesn't exist")
    feat_vectors = list()
    for name in os.listdir(imgs_dir):
        feat_vector = read_img_and_compute_feat_vector(
            os.path.join(imgs_dir, name), attribute_extractor)
        feat_vectors.append(feat_vector)
    return feat_vectors


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def write_gallery_imgs(imgs, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(path, "{:5d}.jpg".format(i)), img)

def calculate_dists(uniqvect, gallery):
    dists = [
        np.average(np.dot(uniqvect, np.transpose(feat_vector)))
        for feat_vector in gallery._feats
    ]
    return dists

# return value is array of indexes of tracks that did not get ided anything
def single_assignment(kb_tracks, gallery):
    max_dists_size = len(gallery._feats)
    if max_dists_size == 0:
        return [i for i in range(len(kb_tracks))]
    assignment_cost_matrix = list()
    for trk in kb_tracks:
        if len(trk.prev_dists) != max_dists_size:
            trk.prev_dists = calculate_dists(trk.prev_feat, gallery)
        assignment_cost_matrix.append(trk.prev_dists)
    rowind, colind = linear_sum_assignment(assignment_cost_matrix)
    for i, ri in enumerate(rowind):
        kb_tracks[ri].reid.append(colind[i])

    if len(rowind) < len(kb_tracks):
        return [i for i in range(len(kb_tracks))
                if i not in rowind]



def run_mot_and_fill_gallery(video_loader, gallery, detector, sort_trackers, attribute_extractor,
                             output_files):

    # Iterate through frames of all cameras
    for findex, frames in tqdm(video_loader):

        # Send frames from each camera to gallery to decide if references need to be captured based off triggering
        gallery.update(frames)

        # Iterate through each camera
        for vidname, frame in frames.items():
            # Get bounding boxes of all people
            boxes, scores = detector.get_bboxes(frame)

            # Send people bounding boxes to tracker
            # Get three things: normal Sort output (tracking bounding boxes it wants to send), corresponding track objects, and objects of new tracks
            tracker = sort_trackers[vidname]
            dets = np.column_stack((np.reshape(boxes, [-1, 4]), scores))
            matched_tracks, matched_kb_trackers, new_kb_trackers = tracker.update(
                dets)

            # Find indexes of returned bounding boxes that meet ideal ratio
            trkbboxes = np.array(matched_tracks)
            widths = trkbboxes[:, 2] - trkbboxes[:, 0]
            heights = trkbboxes[:, 3] - trkbboxes[:, 1]
            aspectratio = heights / widths
            readybools = np.isclose(aspectratio, 2, rtol=0.25)
            indexes = np.arange(len(matched_tracks))[readybools]

            for index in indexes:
                trk = matched_tracks[index]
                box = ((int(trk[0]), int(trk[1])), (int(trk[2]), int(trk[3])))
                cropimg = crop_image(frame, box)
                if cropimg.size <= 5: continue
                uniqvect = attribute_extractor(cropimg)
                dists = calculate_dists(uniqvect, gallery)
                matched_kb_trackers[index].prev_feat = uniqvect
                matched_kb_trackers[index].prev_dists = dists

            # row index is track, column index is track id
            not_assigned = list()
            if len(indexes) != 0:
                not_assigned = single_assignment(matched_kb_trackers, gallery)


            # Iterate through returned bounding boxes
            for ind, trk in enumerate(matched_tracks):
                box = ((int(trk[0]), int(trk[1])), (int(trk[2]), int(trk[3])))

                # Write bounding box, frame number, and trackid to file
                if ind in not_assigned:
                    reidmode = -1
                else:
                    reidmode = mode(matched_kb_trackers[ind].reid)[0][0]
                output_files[vidname].write("%d,%d,%.2f,%.2f,%.2f,%.2f\n" %
                                            (findex, reidmode, box[0][0],
                                             box[0][1], box[1][0], box[1][1]))

            # Iterate through new tracks and add their current bounding box to list of track references
            for trk in new_kb_trackers:
                d = trk.get_state()[0]
                box = ((int(d[0]), int(d[1])), (int(d[2]), int(d[3])))
                cropimg = crop_image(frame, box)
                if cropimg.size > 5: continue
                trk.prev_feat = attribute_extractor(cropimg)
                trk.prev_dists = calculate_dists(trk.prev_feat, gallery)
            single_assignment(new_kb_trackers, gallery)


def convert_files_to_numpy(temp_dir, output_files):
    for video_name, file_handle in output_files.items():
        file_handle.close()
        output_files[video_name] = np.loadtxt(os.path.join(
            temp_dir, "{}.txt".format(video_name)),
                                              delimiter=',')


def main():
    args = init_args()

    detector = FasterRCNN()
    attribute_extractor = MgnWrapper(args.weights_path)
    dataloader = loaders.get_loader(args.video_path, args.loader,
                                    args.interval)

    ref_img = cv2.imread(args.ref_image_path)
    # TODO: (nhendy) do this mapping in a config file
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
    sort_trackers = {
        vidnames: Sort()
        for vidnames in dataloader.get_vid_names()
    }
    output_files = {
        vidnames: open(os.path.join("{}.txt".format(vidnames)), "w")
        for vidnames in dataloader.get_vid_names()
    }

    # Run detector, Sort and fill up gallery
    run_mot_and_fill_gallery(dataloader, gallery, detector, sort_trackers, attribute_extractor,
                             output_files)

    # Save images from gallery captured throughout video
    write_gallery_imgs(gallery.people, args.gallery_path)


if __name__ == "__main__":
    sys.exit(main())
