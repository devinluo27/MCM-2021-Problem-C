from PIL import Image
import os
import torch
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


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

class MyDataset(Dataset):
    def __init__(self, root_dir, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0], int(words[1])))
        
        self.imgs = imgs
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(self.root_dir+fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(root_dir=train_dir,datatxt=id_label, transform=data_transforms['train'])
# test_data=MyDataset(root_dir=val_dir,datatxt=, transform=data_transforms['val'])


sample_weights = [5 if label == 1 else 1 for _, label in train_data]
train_sampler = WeightedRandomSampler(sample_weights, len(train_data), replacement=True)
trainset_dataloader = DataLoader(dataset=train_data,
                                 batch_size=4,
                                 num_workers=4, sampler=train_sampler)
print('weighted random dataloader finished!')

if __name__ == '__main__':
        
    cnt = 0
    to_pil_image = transforms.ToPILImage()
    for image,label in trainset_dataloader:
        if cnt>=3:      # 只显示3张图片
            break
        print(label)    # 显示label

        # 方法1：Image.show()
        # transforms.ToPILImage()中有一句
        # npimg = np.transpose(pic.numpy(), (1, 2, 0))
        # 因此pic只能是3-D Tensor，所以要用image[0]消去batch那一维
        img = to_pil_image(image[0])
        img.show()

        # 方法2：plt.imshow(ndarray)
        # img = image[0]      # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
        # img = img.numpy()   # FloatTensor转为ndarray
        # img = np.transpose(img, (1,2,0))    # 把channel那一维放到最后
        # # 显示图片
        # plt.imshow(img)
        # plt.show()

        cnt += 1
