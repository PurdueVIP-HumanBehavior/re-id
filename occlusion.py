import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  #os.sys
from PIL import Image

import torch
import torch.nn as nn
import torchvision

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision import transforms

from attribute_extractors import MgnWrapper

# model -> MGN - deep learning model
# image -> b_box cropped image of the person
# label -> MGN output label for the image


def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0.5):

    #get the width and height of the img
    width, height = image.shape[-1], image.shape[-2]

    #set the output img width and height
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))

    #create a white image with the sizes defined above
    heatmap = torch.zeros((output_height, output_width))

    #iterate over all the pixels
    for h in range(0, height):
        for w in range(0, width):

            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if (w_end) >= width or (h_end) >= height:
                continue

            input_image = image.clone().detach()

            #replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, h_start:h_end, w_start:w_end] = occ_pixel

            #since MGN doesn't accept tensor, convert input_image to PIL
            input_image = transforms.ToPILImage()(input_image)

            #run inference on modified image
            output = model(input_image)

            #get MGN output for the image
            output = nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][label]

            #setting the heatmap location to probability value
            heatmap[h, w] = prob

    return heatmap


#import MGN
attribute_extractor = MgnWrapper("./model.pt")

#get image and cast it to Tensor
jpegfile = Image.open(
    "occ_analysis/amogh/1.jpg")  #change the input file location accordingly
image = ToTensor()(jpegfile)  #.unsqueeze(0)
image = Variable(image)

#get dimensions of the input img for debug
width, height = jpegfile.size

#get the output by using original img
mgn_output_orig = attribute_extractor(jpegfile)

#pass the outputs through softmax to interpret them as probability
outputs = nn.functional.softmax(mgn_output_orig, dim=1)

#get the maximum predicted label
prob_no_occ, pred = torch.max(outputs.data, 1)

#get the first item
prob_no_occ = prob_no_occ[0].item()

#run occ analysis and display heatmap
heatmap = occlusion(attribute_extractor, image, pred[0].item(), 10, 7)
imgplot = sns.heatmap(heatmap,
                      xticklabels=False,
                      yticklabels=False,
                      vmax=prob_no_occ)
figure = imgplot.get_figure()
figure.savefig('./occ_analysis/amogh/heatmap/amoghout1-try.png',
               dpi=400)  #change the output file location
