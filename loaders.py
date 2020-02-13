from constants import defaultkey
from PIL import Image
import os
import cv2


def get_loader(path, typeloader, interval):
    # check if path exists
    if not os.path.exists(path):
        raise ValueError("path: {} does not exist".format(path))
    contents = os.listdir(path)

    # check if path is empty
    if len(contents) == 0:
        raise ValueError("nothing in {}".format(path))

    contents = [
        name for name in contents
        if not os.path.isdir(os.path.join(path, name))
    ]
    return VideoLoader(path, contents, interval=interval)


class Loader:
    def __init__(self):
        pass

    def get_vid_names(self):
        return self.videos.keys()

      
class VideoLoader(Loader):
    def __init__(self, path, vids, interval=1):
        super().__init__()
        self.path = path
        names = ['.'.join(key.split('.')[:-1]) for key in vids]
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

