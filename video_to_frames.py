import cv2
import os
import sys
import argparse
from tqdm import tqdm

def init_args():
    parser = argparse.ArgumentParser(description="paints videos with bounding boxes and IDs")
    parser.add_argument("source_vid", metavar='src', type=str,
                        help="the source video to use")
    parser.add_argument("dest_dir", metavar="dir", type=str,
                        help="the name of the destination directory")
    return parser.parse_args()

def main():
    args = init_args() 
    input_video_name = args.source_vid 
    output_directory = args.dest_dir
    if not os.path.exists(input_video_name):
        raise ValueError("input video path does not exist")
    cap = cv2.VideoCapture(input_video_name)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    if cap.isOpened() == False:
        print("error openning file")

    i = 0
    numframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if cap.isOpened():
        for i in tqdm(range(numframes)): 
            ret, frame = cap.read()
            if ret:
                namenum = '%09d' % i
                cv2.imwrite(os.path.join(output_directory, namenum + '.jpg'), frame)
                i = i + 1
            else:
                continue


if __name__ == "__main__":
    sys.exit(main())
