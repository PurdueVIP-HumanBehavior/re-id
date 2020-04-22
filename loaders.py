from constants import defaultkey
from PIL import Image
import os
import cv2
import numpy as np


def get_loader(path, typeloader, interval):
    # check if path exists
    if not os.path.exists(path):
        raise ValueError("path: {} does not exist".format(path))
    contents = os.listdir(path)

    # check if path is empty
    if len(contents) == 0:
        raise ValueError("nothing in {}".format(path))

    if typeloader == "videos":
        contents = [
            name for name in contents
            if not os.path.isdir(os.path.join(path, name))
        ]
        return VideoLoader(path, contents, interval=interval)
    elif typeloader == "frames":
        contents = [
            name for name in contents
            if os.path.isdir(os.path.join(path, name))
        ]
        return FrameLoader(path, contents, interval=interval)

class Loader:
    def __init__(self):
        pass

    def get_vid_names(self):
        return self.videos.keys()

class VideoLoader(Loader):
    def __init__(self, path, vids, interval=1):
        super().__init__()
        self.path = path
        names = [os.path.splitext(key)[0] for key in vids]
        self.videos = {
            name: cv2.VideoCapture(os.path.join(path, file))
            for name, file in zip(names, vids)
        }
        self.index = 0
        self.length = min([
            vid.get(cv2.CAP_PROP_FRAME_COUNT)
            for name, vid in self.videos.items()
        ])
        self.interval = interval

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        retval = dict()
        for name, vid in self.videos.items():
            vid.set(cv2.CAP_PROP_POS_FRAMES, self.index)
            success, retval[name] = vid.read()
            if not success:
                raise StopIteration
        indtosend = self.index
        self.index = self.index + self.interval
        return indtosend, retval

    def __len__(self):
        return int(self.length / self.interval)

class FrameLoader(Loader):
    def __init__(self, path, dirs, interval=1):
        super().__init__()
        self.path = path
        # dirs = [os.path.join(path, name) for name in dirs]
        if not isinstance(dirs, list):
            raise TypeError("dirs must be a list of directorys")
        self.videos = {name: sorted(os.listdir(os.path.join(path, name))) for name in dirs} 
        self.index = 0
        self.length = min([
            len(vid)
            for name, vid in self.videos.items()
        ])
        self.interval = interval

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration

        retval = dict()
        for name, imgs in self.videos.items():
            pilimg = Image.open(os.path.join(self.path, name, imgs[self.index]))
            # convert to cv2 numpy format
            retval[name] = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

        indtosend = self.index
        self.index = self.index + self.interval
        return indtosend, retval

    def __len__(self):
        return int(self.length / self.interval)
