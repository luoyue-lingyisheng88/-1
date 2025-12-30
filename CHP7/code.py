import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

color = "tab:red"

ax1.set_xlabel("Epochs")

ax1.set_ylabel("Accuracy", color=color)

ax1.plot(range(1, epochs + 1), accs, color=color, label="Accuracy")

ax1.tick_params(axis="y", labelcolor=color)

ax1.legend(loc="upper left")

ax2 = ax1.twinx()

color = "tab:blue"

ax2.set_ylabel("Loss", color=color)

ax2.plot(range(1, epochs + 1), losses1, color=color, label="Loss")

ax2.tick_params(axis="y", labelcolor=color)

ax2.legend(loc="upper right")

plt.title("Model Accuracy and Loss per Epoch")

fig.tight_layout() # 调整布局以适应第二个y轴

plt.show()

data = datasets.FashionMNIST(
    "data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize((96, 96)), transforms.ToTensor()]
    ),
)

classes = data.classes

model.eval()

testloader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize((96, 96)), transforms.ToTensor()]
        ),
    ),
    batch_size=20,
    shuffle=True,
)

dataiter = iter(testloader)

images, labels = next(dataiter)

images = images.to(device)

outputs = model(images)

predicted = torch.max(outputs, 1)

plt.figure(figsize=(10, 10))

for i in range(20):
    plt.subplot(5, 4, i + 1)
    plt.imshow(images[i].cpu().squeeze(), cmap="gray")  # 显示图片，确保转换回 CPU
    plt.title(f"Predicted: {classes[predicted[i].item()]}")
    plt.axis("off")
plt.show()
