import torch
import torch.nn as nn
import os
from mobile import Net
# optimizer
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.mobilenet import mobilenet_v2
from MyDataset import MyDataset
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICE"] = "0, 1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization for training
# Just normalization for validation
input_size = 224
data_transforms = {
    'train': transforms.Compose([
        # nn.Upsample
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

def train(model, optimizer, loss_fn, num_epoch, data_loader, device):
    running_loss = 0
    for epoch in range(num_epoch):
        for data, labels in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('%d epoch: %f'%(epoch+1, loss))
            break


def evaluate(net, data_loader, device):
    net.eval() #进入模型评估模式
    correct = 0
    total = 0
    predicted_list=[]
    true_list=[]
    with torch.no_grad() :
        for data in data_loader:
            images,labels = data[0].to(device),data[1].to(device)
            true_list = np.append(true_list, labels.numpy())
            outputs = net(images)
            predicted = torch.argmax(outputs.data, 1)
            predicted_list = np.append(predicted_list,predicted.numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total         
    C = confusion_matrix(true_list, predicted_list)
    return acc, C
 
def show_confMat(confusion_mat, classes_name, set_name, out_dir):
    """
    可视化混淆矩阵，保存png格式
    :param confusion_mat: nd-array
    :param classes_name: list,各类别名称
    :param set_name: str, eg: 'valid', 'train'
    :param out_dir: str, png输出的文件夹
    :return:
    """
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes_name)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()
 
    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    # plt.colorbar()
 
    # 设置文字
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=60)
    plt.yticks(xlocations, classes_name)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)
 
    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.png'))
    plt.close()


if __name__ == '__main__':

    # create a dataloader
    batch_size = 16
    root = os.getcwd()
    train_dir = root + '/train/'
    val_dir = root + '/val/'
    id_label = root + '/id_label_jpg.csv'
    train_data = MyDataset(root_dir=train_dir, datatxt=id_label, transform=data_transforms['train'])
    sample_weights = [5 if label == 1 else 1 for _, label in train_data]
    train_sampler = WeightedRandomSampler(sample_weights, batch_size, replacement=True)
    trainset_dataloader = DataLoader(dataset=train_data,
                                    batch_size=4,
                                    # shuffle=True,
                                    num_workers=4, sampler=train_sampler)
    print('weighted random dataloader finished!')


    #############
    # hyper params
    num_epoch = 1
    learning_rate = 0.001

    # create my model
    model = models.mobilenet_v2(pretrained=True)
    model = Net(model).to(device)
    model = model.to(device)
    loss_weight = torch.FloatTensor([1, 5]).to(device)
    loss = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = optim.rmsprop(model.parameters(), lr=learning_rate)

    train(model=model, optimizer=optimizer, loss_fn=loss, 
            num_epoch=num_epoch,data_loader=train_data,device=device)
    
    train_acc, C1 = evaluate(model, train_data, device)
    print('Training Accuracy: %.2f%%'% (100 * train_acc))

