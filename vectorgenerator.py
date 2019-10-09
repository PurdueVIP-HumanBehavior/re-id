import opt

from MGN import MGN
import torch
from torch.autograd import Variable
from torchvision import transforms

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
        person = person.cuda()  #assumes that you're using GPU
        person = self.model(person)
        return person


options = {
    opt.defaultkey: MGN_Wrap,
    "MGN": MGN_Wrap
}