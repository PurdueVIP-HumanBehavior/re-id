class ResNet50_nFC_Wrap:
    def __init__(self, class_num, weights_path):
        self.model = ResNet50_nFC(class_num)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.cuda()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((288, 144)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def getVect(self, person):
        person = self.transform(person).float()
        person = person.unsqueeze(dim=0)
        person = Variable(person, requires_grad=True)
        person = person.cuda()  #assumes that you're using GPU
        person = self.model(person)
        return person


num_cls_dict = {'market': 30, 'duke': 23}


class TripleNet:
    def __init__(self):
        self.model1 = ResNet50_nFC_Wrap(30, 'market_attr_net_last.pth')
        self.model2 = ResNet50_nFC_Wrap(23, 'duke_attr_net_last.pth')
        self.model3 = MgnWrapper()

    def getVect(self, person):
        vec1 = self.model1.getVect(person)
        vec2 = self.model2.getVect(person)
        vec3 = self.model3.getVect2(person)
        print(vec1.shape)
        print(vec2.shape)
        print(vec3.shape)

    def getVect2(self, person):
        self.getVect(person)
