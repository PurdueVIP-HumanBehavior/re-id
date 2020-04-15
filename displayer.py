import cv2
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm


FRAME_INDEX = 0
ID_INDEX = 1
X1_INDEX = 2
Y1_INDEX = 3
X2_INDEX = 4
Y2_INDEX = 5

OUT_VIDEO_CODEC = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
OUT_VIDEO_EXTENSION = ".mp4"

def main():
    args = init_args()
    create_vid(args.dest_vid, args.source_vid, args.ids_txt, view=args.view, delimiter=args.delimiter)


def init_args():
    parser = argparse.ArgumentParser(description="paints videos with bounding boxes and IDs")
    parser.add_argument("-v", "--view", default=False, action='store_true',
                        help="flag used to view the output video as it is being created (is slow)")
    parser.add_argument("-d", "--delimiter", type=str, default=' ',
                        help="the delimiter to use for the bounding box text file")
    parser.add_argument("source_vid", metavar='src', type=str,
                        help="the source video to use")
    parser.add_argument("ids_txt", metavar='ids', type=str,
                        help="the text file of format <frame, id, x1, y1, x2, y2>")
    parser.add_argument("dest_vid", metavar="dest", type=str,
                        help="the name of the destination video")
    return parser.parse_args()


def create_vid(output_video_path, input_video_path, output_text_path, view=False, delimiter=' '):
    """
    Creates our "final reid video" with bounding boxes and ID number over people in the original video

    Parameters:
    output_video_path (str): the filename to save the video as
    input_video_path (str): the video file path
    output_text_path(str): the output text file

    Returns:
    1 for successful
    0 for failure
    """

    if not os.path.exists(input_video_path):
        raise ValueError("vid " + input_video_path + " does not exist")

    input_video = cv2.VideoCapture(input_video_path)
    if input_video.isOpened() == False:
        raise RuntimeError("error opening file " + input_video_path)

    if not os.path.exists(output_text_path):
        raise ValueError("outtxt " + output_text_path + "does not exist")

    frmame_id_data = np.loadtxt(output_text_path, delimiter=delimiter)
    if frmame_id_data.shape[1] != 6:
        raise ValueError("The text file should have 6 entries per row. yours has {}".format(frmame_id_data.shape[1]))

    frame_indexes = np.sort(np.unique(frmame_id_data[:, FRAME_INDEX]))
    interval = np.average(frame_indexes[1:] - frame_indexes[:-1]).astype(np.int64)

    output_video_path = append_extension(output_video_path)
    while os.path.exists(output_video_path):
        decision = input("the video file " + output_video_path + " already exists. Want to overwrite it? [y/n] ").lower()
        if decision == 'n':
            output_video_path = input("new file name: ")
            output_video_path = append_extension(output_video_path)
        else:
            break

    video_fps = int(input_video.get(cv2.CAP_PROP_FPS) / interval)
    video_size = (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_video = cv2.VideoWriter(output_video_path,
                             OUT_VIDEO_CODEC,
                             video_fps, video_size)

    for frame in tqdm(frame_indexes):
        frame_rows = frmame_id_data[frmame_id_data[:, FRAME_INDEX] == frame, :]
        input_video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = input_video.read()
        if ret:
            bboxes = frame_rows[:, X1_INDEX:(Y2_INDEX + 1)].astype(np.int64)
            ids = frame_rows[:, ID_INDEX].astype(np.int64)
            nimg = paint_frame(img, bboxes, ids)
            if view:
                cv2.imshow("savename", nimg)
                cv2.waitKey(int(1000 / video_fps))
            output_video.write(nimg)
        else:
            break
    output_video.release()

def paint_frame(img, bboxes, ids):
    """
    paint an image with a list of bounding boxes and associated ids

    Parameters
    img (np.array): the frame as a numpy array
    bboxes (np.array): a list of bouding boxes - numpy array n x 4 array - each row is of form x1,y1,x2,y2
    ids (list): a list of ids for the corresponding bouding boxes

    Returns
    img (np.array): numpy array of image with bounding boxes painted on
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

def append_extension(filename):
    """ Returns the filename with the proper extension"""
    return filename + OUT_VIDEO_EXTENSION

if __name__ == "__main__":
    sys.exit(main())
