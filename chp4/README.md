                                                                线性回归_感知机

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

<img width="1227" height="904" alt="e3d87e8b493efa816a6009f00f140f8" src="https://github.com/user-attachments/assets/d946434b-af60-4e4a-a5c8-a124631825e3" />

2. 模型设计

构建一个简单的多层感知机（MLP）模型，包含输入层、隐藏层和输出层：

输入层：将 28×28 的图像展平为 784 维向量；

隐藏层：使用 128 个神经元，激活函数为 ReLU；

输出层：使用 10 个神经元（对应 10 个类别），无激活函数（因损失函数cross_entropy已内置 Softmax）。

<img width="1042" height="832" alt="79725796e9b38c8251e62c4e5942eef" src="https://github.com/user-attachments/assets/1d071421-5449-406a-bdde-7ecc940ec8f5" />

3. 模型训练与评估

训练阶段：使用随机梯度下降（SGD）优化器，学习率lr=0.1，共训练 10 个 epoch。每次迭代先清空梯度，再通过反向传播更新模型参数。

评估阶段：在测试集上计算损失和准确率，禁用梯度计算以提高效率。

<img width="1055" height="507" alt="11b63452dc2ffb98909e55c56512d5d" src="https://github.com/user-attachments/assets/46c14a7c-4eb9-4bbe-8143-011f7d683024" />

四、实验结果与分析

<img width="1656" height="445" alt="5051e5f5888f0da4c5e2e1b54efff1b" src="https://github.com/user-attachments/assets/71faf515-ab69-43af-a4be-d4ed6cb32606" />

随着训练轮次增加，测试损失逐渐降低，准确率持续上升，说明模型在不断学习并拟合数据。
最终测试准确率达到 90.44%，对于仅含一层隐藏层的简单 MLP 模型，该结果较为合理。若需进一步提升性能，可考虑增加网络层数、调整学习率、引入正则化（如 Dropout）等方法。

五、实验总结

本实验通过 PyTorch 框架实现了基于 MLP 的 MNIST 手写数字识别，完整走通了 “数据集加载 - 模型构建 - 训练 - 评估” 的深度学习流程。实验结果表明，简单的 MLP 模型即可在 MNIST 数据集上取得
较好的识别效果，同时也为后续更复杂模型（如 CNN）的学习奠定了基础
  
