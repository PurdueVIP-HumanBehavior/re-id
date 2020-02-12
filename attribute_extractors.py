from MGN import MGN
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import os
from constants import *


def ndarraytopil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


# TODO: (nhendy) link to the paper
class MgnWrapper:
    def __init__(self, weights_path):
        if not os.path.exists(weights_path):
            raise ValueError(
                "Weights path given {} doesn't exist".format(weights_path))
        self.model = MGN()
        self.model.load_state_dict(torch.load(weights_path))
        self.model.cuda()
        self.model.eval()
        self.transform = transforms.Compose([
            # TODO: (nhendy) this was using bicubic interpolation before. Ask moiz why
            transforms.Resize(INPUT_RESOLUTION, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=PER_CHANNEL_MEAN, std=PER_CHANNEL_STD)
        ])

    def compute_feat_vector(self, inputs):
        # TODO: (nhendy) needs more comments
        if isinstance(inputs, np.ndarray):
            inputs = ndarraytopil(inputs)
        inputs = self.transform(inputs).float()
        ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
        inputs = inputs.unsqueeze(0)
        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(
                    3,
                    torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to('cuda')
            # input_img = input_img.unsqueeze(0)
            outputs = self.model(input_img)
            f = outputs[0].data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        return ff
