import opt

from MGN import MGN
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch import nn
from torch.nn import init
from torchvision import models
from ResNet50_nFC import ResNet50_nFC

num_classes = 751  # change this depend on your dataset

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

    def getVect(self, person):
        person = self.transform(person).float()
        person = Variable(person, requires_grad=True)
        person = person.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        person = person.cuda()  # assumes that you're using GPU
        person = self.model(person)
        return person

    def getVect2(self, inputs):
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
        self.load_state_dict(torch.load(weights_path))
        self.model.cuda()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((288, 144)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def getVect(self, person):
        person = self.transform(person).float()
        person = person.unsqueeze(dim=0)
        person = Variable(person, requires_grad=True)
        person = person.cuda()  #assumes that you're using GPU
        person = self.model(person)
        return person


options = {
    opt.defaultkey: MGN_Wrap,
    "MGN": MGN_Wrap,
    'ResNet50_nFC':ResNet50_nFC_Wrap
}