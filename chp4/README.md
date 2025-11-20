一、实验目的
MNIST 数据集是手写数字识别领域的经典数据集，包含 60000 张训练图像和 10000 张测试图像，每张图像为 28×28 像素的灰度图，对应 0-9 共 10 个类别。本实验旨在通过构建多层感知机（MLP）模型，实现对 MNIST 手写数字的识别，熟悉深度学习框架 PyTorch 的使用流程，理解神经网络训练与评估的基本原理。
二、实验环境
操作系统：Windows
开发框架：PyTorch
编程语言：Python
主要库：torch、torchvision、numpy、matplotlib
三、实验方法与步骤
1. 数据集加载与预处理
使用torchvision库加载 MNIST 数据集，并通过DataLoader按批次（batch_size=2048）加载数据，同时对图像进行tensor转换。

  
