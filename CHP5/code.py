import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设置批处理大小
batch_size = 512

# 检查CUDA设备是否可用，并设置为设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据的转换方式，这里仅将数据转换为tensor
transform = transforms.Compose([transforms.ToTensor()])

# 加载训练集
train_loader = DataLoader(
    datasets.MNIST(root='data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# 加载测试集
test_loader = DataLoader(
    datasets.MNIST(root='data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 定义全连接层
        self.fc1 = nn.Linear(5*5*16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # 定义分类层
        self.clf = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.sigmoid(x)

        x = self.fc2(x)
        x = F.sigmoid(x)

        x = self.clf(x)
        return x

model = ConvNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-2)
model

epochs = 30
accs, losses = [], []

for epoch in range(epochs):
    # 遍历训练集
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        # 计算损失并反向传播
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    correct = 0
    testloss = 0
    # 在测试集上进行评估
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            # 计算测试集上的损失
            testloss += F.cross_entropy(out, y).item()
            # 计算准确率
            pred = out.max(dim=1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    # 计算平均测试集损失和准确率
    acc = correct / len(test_loader.dataset)
    testloss /= (batch_idx + 1)
    accs.append(acc)
    losses.append(testloss)

    print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(epoch, testloss, acc))

# 绘制准确率/损失曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # 一行两列，第一个子图
plt.plot(accs, label='Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.subplot(1, 2, 2)  # 一行两列，第二个子图
plt.plot(losses, label='Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.tight_layout()
plt.show()

# 获取模型各卷积层的特征图（需确保此时x是测试集中的一个batch数据）
feature1 = model.conv1(x)
feature2 = model.conv2(feature1)
n = 5

img = x.detach().cpu().numpy()[:n]
feature_map1 = feature1.detach().cpu().numpy()[:n]
feature_map2 = feature2.detach().cpu().numpy()[:n]

fig, ax = plt.subplots(nrows=3, n, figsize=(10, 10))
for i in range(n):
    # 对输入图像进行求和以便在灰度图中显示
    ax[0, i].imshow(img[i].sum(0), cmap='gray')
    ax[0, i].set_title(f'Input Image {i + 1}')
    ax[1, i].imshow(feature_map1[i].sum(0), cmap='gray')
    ax[1, i].set_title(f'Feature Map 1.{i + 1}')
    ax[2, i].imshow(feature_map2[i].sum(0), cmap='gray')
    ax[2, i].set_title(f'Feature Map 2.{i + 1}')

# 调整子图间距
plt.tight_layout()
plt.show()
