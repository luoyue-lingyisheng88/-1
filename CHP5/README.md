                                                                   实验五 LeNet

一 实验目的

学会用 PyTorch 设计LeNet网络结构，并且以 MNIST 数据集为对象进行对LeNet模型设计、数据加载、损失函数及优化器定义和验证评估模型的性能。

二 实验内容

导入所需的依赖包

<img width="553" height="222" alt="image" src="https://github.com/user-attachments/assets/6f9ddb1a-1b69-4cda-9a07-46cb2c0aa25c" />

导入训练数据

<img width="1020" height="574" alt="image" src="https://github.com/user-attachments/assets/8f116973-407d-4c69-a8ea-120c2fc94bd3" />

定义模型

<img width="1090" height="867" alt="image" src="https://github.com/user-attachments/assets/574efa4e-b7bd-405c-b7ef-6908bbd06f69" />

模型初始化

<img width="592" height="87" alt="image" src="https://github.com/user-attachments/assets/be2e2589-8bf7-4486-a4ee-a34a2ce2c2c3" />

模型训练

<img width="783" height="720" alt="image" src="https://github.com/user-attachments/assets/1d121b55-1083-4a1e-8f00-a30f7ac52991" />


<img width="1104" height="720" alt="image" src="https://github.com/user-attachments/assets/3cb43579-7d89-45c9-af6c-c0bc521595f3" />


<img width="278" height="240" alt="4ff51217bf3f6e6f61828cb5209f739" src="https://github.com/user-attachments/assets/044bf133-7e86-4ab9-b990-95a35e95506a" />

acc与loss变化趋势


<img width="514" height="302" alt="ed53cc2a8081f9ba0f15fb313b3379d" src="https://github.com/user-attachments/assets/ac395150-5393-4a50-a872-424a2e396a2b" />


<img width="1047" height="582" alt="image" src="https://github.com/user-attachments/assets/12a3cf83-6be6-4c85-bf16-d93cfad01fec" />

特征图可视化


<img width="461" height="371" alt="9e085345e9d313e1f0160868502aa7a" src="https://github.com/user-attachments/assets/ad4ed5e8-4423-4a99-a990-5ff84910a067" />


<img width="1281" height="960" alt="image" src="https://github.com/user-attachments/assets/becc2633-4c4b-4b96-9395-45c77ae45da8" />

三 实验总结

本次实验基于 PyTorch 框架搭建 LeNet网络，在 MNIST 数据集上完成训练与测试。实验先导入相关库、配置数据预处理与加载方案，再通过 ConvNet 类定义网络的卷积、池化与全连接层结构。训练阶段采用优化器和交叉熵损失函数，经 30 个 epoch 训练后，模型准确率明显提升、损失持续下降。通过绘制准确率与损失曲线，直观验证了训练的有效性，结合特征图可视化，清晰观察到网络各层的特征提取规律。整个实验提升了我对深度学习模型构建、训练及评估的实践能力。



