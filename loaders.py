import opt
from PIL import Image
import os

def getLoader(path):
    # check if path exists
    if not os.path.exists(path):
        raise ValueError("path: {} does not exist".format(path))
    contents = os.listdir(path)

    # check if path is empty
    if len(contents) == 0:
        raise ValueError("nothing in {}".format(path))

    # if working with frames create loader with directories
    if options[opt.args.loader] == "frames":
        frames = {cont: os.listdir("{}/{}".format(path, cont)) for cont in contents}

        # the least number of frames a video has
        minlen = min([len(cont) for cont in frames.values()])

        #clipping all videos to match
        frames = {key: cont[0: minlen] for key, cont in frames.items()}
        return FrameLoader(path, frames)


class FrameLoader:
    def __init__(self, path, frames):
        self.path = path
        self.videos = frames
        self.index = 0
        self.length = len(list(frames.values())[0])

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        frames = {key: frames[self.index] for key, frames in self.videos.items()}
        frames = {key: Image.open(os.path.join(self.path, key, img)) for key, img in frames.items()}
        self.index = self.index + opt.args.interval
        return self.index, frames

options = {
    opt.defaultkey: "frames",
    "frames": "frames"
}

