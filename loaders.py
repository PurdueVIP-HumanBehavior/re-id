from opt import defaultkey
import os

def getLoader(path, option):
    # check if path exists
    if not os.path.exists(path):
        raise ValueError("path: {} does not exist".format(path))
    contents = os.listdir(path)

    # check if path is empty
    if len(contents) == 0:
        raise ValueError("nothing in {}".format(path))

    # if working with frames create loader with directories
    if option == options["frames"]:
        frames = {cont: os.listdir("{}/{}".format(path, cont)) for cont in contents}

        # the least number of frames a video has
        minlen = min([len(cont) for cont in frames.values()])

        #clipping all videos to match
        frames = {key: cont[0: minlen] for key, cont in frames.items()}
        return FrameLoader(frames)


class FrameLoader:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):

        self.index = self.index + 1

options = {
    defaultkey: "frames"
}

