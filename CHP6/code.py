import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchsummary import summary

batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    ),
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    ),
    batch_size=batch_size,
    shuffle=False
)


def get_model(name, pretrained):
    if name == "v1":
        model = models.GoogLeNet(weights=pretrained)
    elif name == "v3":
        model = models.inception_v3(weights=pretrained)
    return model


class Inception(nn.Module):
    def __init__(self, name, pretrained, classes):
        super(Inception, self).__init__()
        self.base = get_model(name, pretrained)
        self.base.fc = nn.Linear(self.base.fc.in_features, classes)

    def forward(self, x):
        output = self.base(x)
        if isinstance(output, torch.Tensor):
            return output
        else:
            return output.logits


classes = 10
name = "v3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Inception(name, pretrained=True, classes=classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
summary(model, input_size=(3, 299, 299))

# 训练循环配置
epochs = 10
accs, losses = [], []  # 存储准确率与测试损失

for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    # 训练阶段
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()  # 清除之前的梯度
        x, y = x.to(device), y.to(device)  # 将数据移动到设备（CPU/GPU）
        out = model(x)  # 前向传播
        loss = F.cross_entropy(out, y)  # 计算训练损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

    # 初始化测试统计信息
    correct = 0
    testloss = 0.0
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        # 测试阶段
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)  # 数据移送至设备
            out = model(x)  # 前向传播
            testloss += F.cross_entropy(out, y).item()
            pred = out.max(dim=1, keepdim=True)[1]  # 获取预测类别
            correct += pred.eq(y.view_as(pred)).sum().item()  # 统计正确数

    # 计算准确率与测试平均损失
    acc = correct / len(test_loader.dataset)
    avg_testloss = testloss / len(test_loader)
    accs.append(acc)
    losses.append(avg_testloss)  # 存储测试平均损失

    # 打印当前轮结果
    print("Epochs: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, avg_testloss, acc))
