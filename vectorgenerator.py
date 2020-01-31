from constants import defaultkey
from MGN import MGN
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch import nn
from torch.nn import init
from torchvision import models
from ResNet50_nFC import ResNet50_nFC
import numpy as np
import cv2
from PIL import Image

num_classes = 751  # change this depend on your dataset

def ndarray_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

class MGN_Wrap:
    def __init__(self):
        self.model = MGN()
        self.model.load_state_dict(torch.load('model.pt'))
        self.model.cuda()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_vect(self, person):
        person = self.transform(person).float()
        person = Variable(person, requires_grad=True)
        person = person.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        person = person.cuda()  # assumes that you're using GPU
        person = self.model(person)
        return person

    def get_vect2(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = ndarray_to_pil(inputs)
        inputs = self.transform(inputs).float()
        ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
        inputs = inputs.unsqueeze(0)
        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to('cuda')
            # input_img = input_img.unsqueeze(0)
            outputs = self.model(input_img)
            f = outputs[0].data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        return ff


class ResNet50_nFC_Wrap:
    def __init__(self, class_num, weights_path):
        self.model = ResNet50_nFC(class_num)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.cuda()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((288, 144)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_vect(self, person):
        person = self.transform(person).float()
        person = person.unsqueeze(dim=0)
        person = Variable(person, requires_grad=True)
        person = person.cuda()  #assumes that you're using GPU
        person = self.model(person)
        return person


num_cls_dict = { 'market':30, 'duke':23 }


class TripleNet:
    def __init__(self):
        self.model1 = ResNet50_nFC_Wrap(30, 'market_attr_net_last.pth')
        self.model2 = ResNet50_nFC_Wrap(23, 'duke_attr_net_last.pth')
        self.model3 = MGN_Wrap()

    def get_vect(self, person):
        vec1 = self.model1.get_vect(person)
        vec2 = self.model2.get_vect(person)
        vec3 = self.model3.get_vect2(person)
        print(vec1.shape)
        print(vec2.shape)
        print(vec3.shape)
        

    def get_vect2(self, person):
        self.get_vect(person)


options = {
    defaultkey: MGN_Wrap,
    "MGN": MGN_Wrap,
    'ResNet50_nFC': ResNet50_nFC_Wrap,
    'TripleNet': TripleNet
}