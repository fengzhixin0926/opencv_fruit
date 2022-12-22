import torch
from dataloader import *
from torch.utils.data import DataLoader
from model import *
import torch.nn as nn
import torch.optim as optim

train_data = ImageSet(train='train')
test_data = ImageSet(train='test')

# 数据集长度
train_data_len = len(train_data)
test_data_len = len(test_data)

# 利用dataloader加载数据集
batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
print('The length of training:{}\nThe length of test:{}'.format(train_data_len, test_data_len))

# 创建网络模型
model = Model(num_classes=11)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

total_train_step = 0
total_test_step = 0
test_acc = 0

epoch = 100

for i in range(epoch):
    print('------第{}轮训练开始-------'.format(i+1))
    model.trian()
    total_train_loss = 0
    for data in train_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        total_train_loss += loss.item()
    train_loss = total_train_loss / len(train_data)
    print("The loss of training:{}".format(train_loss))

    total_acc = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, tagets = data
            outputs = model(imgs)
            est_labels = torch.argmax(outputs, dim=1)
            acc = (est_labels == tagets).sum().item()
            total_acc += acc
        print('test_accuracy:{}'.format(total_acc / len(test_data)))

        if test_acc < total_acc / test_data_len:
            test_acc = total_acc / test_data_len
            torch.save(model, "F:/OpenCV/best.pth.tar")
            print("model saved")

        print("The best test_acc:{}".format(test_acc))

print('-----The best test_Acc is %f-----' % test_acc)