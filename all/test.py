from dataloader import *
from torch.utils.data import DataLoader
model_path = "F:/OpenCV/best.pth.tar"
model = torch.load(model_path)
test_data = ImageSet(train='test')
batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
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