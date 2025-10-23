                                                               实验 2 ：图像增强

一、 实验目的
学会 Opencv 的基本使用方法，利用 Opencv 等计算机视觉库对图像进行平滑、滤波等操作，实现图像增强。

二、 实验内容
2.1  导入图像滤波相关的依赖包

<img width="462" height="169" alt="image" src="https://github.com/user-attachments/assets/4b23c95b-a6ea-4cb5-9787-84f583a9b030" />

2.2  读取原始图像并进行色彩空间转换

<img width="752" height="377" alt="image" src="https://github.com/user-attachments/assets/933b6ef2-60b7-48cc-8cd7-2357fb0a7eae" />

<img width="953" height="438" alt="image" src="https://github.com/user-attachments/assets/ed54f5ef-982b-419d-903c-b830d5fed9e3" />

<img width="835" height="450" alt="image" src="https://github.com/user-attachments/assets/f852674d-89df-4b38-9667-ffab46ca09ad" />

2.3  添加噪声

<img width="694" height="393" alt="image" src="https://github.com/user-attachments/assets/3c1bf2b2-6b55-4f23-8769-25f580d02636" />

<img width="694" height="362" alt="image" src="https://github.com/user-attachments/assets/65e423a6-8ca0-4771-b8e7-690c79d850bb" />

<img width="694" height="416" alt="image" src="https://github.com/user-attachments/assets/a95f6d6e-4922-47c1-94cd-1b3b403e718a" />

2.4  图像滤波

<img width="694" height="398" alt="image" src="https://github.com/user-attachments/assets/b09b4d14-63a6-4316-b87d-246ca0768aeb" />

<img width="698" height="343" alt="image" src="https://github.com/user-attachments/assets/80fd0d8c-7ac9-48d8-b65e-b76f83b3b11e" />

<img width="757" height="612" alt="image" src="https://github.com/user-attachments/assets/d3c2e1bf-f0c9-4363-9cd2-e14be404cbc8" />

<img width="695" height="604" alt="image" src="https://github.com/user-attachments/assets/460d2463-a663-453a-83ce-aefa11011dd4" />

三、  实验结果与分析

本次图像增强实验围绕OpenCV等工具的应用展开，核心目标是掌握图像平滑与滤波操作，通过实践深入理解图像增强的原理与流程，整体实验达到了预期效果。
实验流程清晰，从基础准备到核心操作逐步推进。首先完成依赖包导入，为后续操作搭建环境；接着读取图像并进行色彩空间转换，解决了OpenCV默认BGR格式与matplotlib RGB显示格式的冲突，还成功将图像转为灰度图，简化后续处理；随后为RGB图像添加椒盐噪声和高斯噪声，模拟实际中受干扰的图像场景；最后重点实现图像滤波，不仅调用API完成基础滤波，还手动编写了均值滤波、中值滤波的核心函数，包括卷积操作、边界填充、中值计算等关键模块，满足了实验对滤波实现方式的要求。
实验过程中也暴露了一些问题，比如初始代码存在语法错误，像函数参数传递格式有误、变量名拼写错误等，通过逐行检查代码逻辑、对照语法规则修正得以解决；在手动实现滤波时，对边界填充模式的选择和卷积计算的矩阵运算理解不够深入，通过查阅资料和调试代码，最终掌握了不同填充模式的适用场景及卷积的计算逻辑。
此次实验意义显著，不仅熟练掌握了OpenCV的基本使用方法，更深入理解了均值滤波、中值滤波等算法的原理。同时，手动编码实现滤波功能，提升了编程能力和问题解决能力，为后续更复杂的计算机视觉任务奠定了坚实基础。
