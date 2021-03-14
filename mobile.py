import torch
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.models.mobilenet import mobilenet_v2
from MyDataset import MyDataset
import os


root = os.getcwd()
train_dir = root + '/train/'
val_dir = root + '/val/'
id_label = root + '/id_label_jpg.csv'
# Data augmentation and normalization for training
# Just normalization for validation
input_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_data=MyDataset(root_dir=train_dir,datatxt=id_label, transform=data_transforms['train'])

trainset_dataloader = DataLoader(dataset=train_data,
                                 batch_size=4,
                                 shuffle=False,
                                 num_workers=4)



#############
model = models.mobilenet_v2(pretrained=True)
mobile_layer = nn.Sequential(*list(model.children()))
# for i, j in model.named_parameters():
#     print(i)
# print(mobile_layer)

class Net(nn.Module):
    #此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
    def __init__(self , model):	
        super(Net, self).__init__()
        self.model = model
        self.linear_1 = nn.Linear(1000, 100)
        self.linear_2 = nn.Linear(100, 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1) 
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    model = Net(model)
    with torch.no_grad():
        for i, j in trainset_dataloader:
            print(i.shape)
            output = model(i)
            print(output)
            print(j)
            break

