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
    def __init__(self, path, interval=1):
        super().__init__()
        self._vid = cv2.VideoCapture(path)
        self._idx = 0
        self._interval = interval

    def __iter__(self):
        while (1):
            self._vid.set(cv2.CAP_PROP_POS_FRAMES, self._idx)
            success, frame = self._vid.read()
            if not success:
                raise StopIteration
            yield frame
            self._idx += self._interval
