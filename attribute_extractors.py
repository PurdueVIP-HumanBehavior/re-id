from third_party.mgn import MGN
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import os
from constants import INPUT_RESOLUTION, PER_CHANNEL_MEAN, PER_CHANNEL_STD


def ndarraytopil(img):
    """Return a PIL image of an ndarray"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)



class MgnWrapper:
    """
    This class is a wrapper class for the attribute extractor MGN

    Attributes:
    model (MGN): MGN model architecture
    transform (Compose): transformations done to an image such as reshaping and normalizing
    """
    def __init__(self, weights_path):
        """
        The constructor for MgnWrapper class

        Parameters:
        weights_path (str): MGN model weights path (MGN.pt)

        """
        if not os.path.exists(weights_path):
            raise ValueError(
                "Weights path given {} doesn't exist".format(weights_path))
        self.model = MGN()
        self.model.load_state_dict(torch.load(weights_path))
        self.model.cuda()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(INPUT_RESOLUTION, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=PER_CHANNEL_MEAN, std=PER_CHANNEL_STD)
        ])

    def compute_feat_vector(self, inputs):
        """
        Uses model to compute the feature vector given an image

        Parameters:
        inputs (Image or ndarray): PIL Image or ndarray of an image

        Returns:
        ndarray: The features extracted from MGN of the given image
        """
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
            outputs = self.model(input_img)
            f = outputs[0].data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        return ff

    def __call__(self, x):
        """ Return the feature vector of an image x """
        return self.compute_feat_vector(x)

