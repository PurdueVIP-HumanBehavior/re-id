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

    # if working with frames create loader with directories
    # if options[typeloader] == "frames":
    #     contents = [
    #         name for name in contents
    #         if os.path.isdir(os.path.join(path, name))
    #     ]
    #     frames = {
    #         cont: os.listdir("{}/{}".format(path, cont))
    #         for cont in contents
    #     }

    #     # the least number of frames a video has
    #     minlen = min([len(cont) for cont in frames.values()])

    #     #clipping all videos to match
    #     frames = {key: cont[0:minlen] for key, cont in frames.items()}
    #     return FrameLoader(path, frames, interval=interval)

    # elif options[typeloader] == "videos":
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
        self.videos = {
            name: cv2.VideoCapture(os.path.join(path, file))
            for name, file in zip(names, vids)
        }
        self._idx = 0
        self._interval = interval

    def __iter__(self):
        while (1):
            vid.set(cv2.CAP_PROP_POS_FRAMES, self._idx)
            succes, frame = vid.read()
            if not succes:
                raise StopIteration
            yield frame
            self._idx += self._interval
