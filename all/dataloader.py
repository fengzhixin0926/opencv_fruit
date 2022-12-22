import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class ImageSet(Dataset):
    def __init__(self, train='train'):
        train_data_count = int(0.75 * 814)   # 训练集数目
        data = np.load("F:/OpenCV/all/fruit.npz", allow_pickle=True)
        # print(data["data_list"])
        self.train_data = data["data_list"][0: int(train_data_count)]
        self.train_label = data["labels"][0: int(train_data_count)]
        self.test_data = data["data_list"][int(train_data_count):]
        self.test_label = data["labels"][int(train_data_count):]
        # print(self.train_label)
        self.train = train
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])

    def __len__(self):
        if self.train == 'train':
            num = self.train_data.shape[0]
            return num
        else:
            num = self.test_data.shape[0]
            return num

    def __getitem__(self, item):
        if self.train == 'train':
            # print(item)
            label = self.train_label[item]
            photo = self.train_data[item]
            photo = self.train_transform(photo)
            return photo, label
        else:
            label = self.test_label[item]
            photo = self.test_data[item]
            photo = self.test_transform(photo)
            return photo, label



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    train = ImageSet()
    batch_size = 4

    data = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)

    dataiter = iter(data)
    photo, label = dataiter.next()

    for i in range(batch_size):
        print(label[i])
        # print(photo)
        imshow(photo[i])






