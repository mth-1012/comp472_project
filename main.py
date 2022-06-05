from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import CNN as CNN


def show_batch(data_loader):
    for images, labels in data_loader:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(torchvision.utils.make_grid(images[:32], nrow=8).permute(1, 2, 0))


def model_evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the {} test images: {} %'
              .format(total, (correct / total) * 100))


def model_save(model):
    torch.save(model.state_dict(), 'model.pt')


def model_evaluate_single(model, image_dir):
    single = Image.open(image_dir).convert('RGB')
    input = transform(single)
    input = input.unsqueeze(0)
    input = input.to(device)
    output = model(input)
    _, predicted = torch.max(output.data, 1)
    result = predicted[0].item()
    print(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('COMP 472 Project')
    print('AI Face Mask Detector')

    """Load dataset"""
    num_epochs = 4
    num_classes = 4
    learning_rate = 0.001

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.25,0.25,0.25), (0.25,0.25,0.25)),
            transforms.CenterCrop(200),
            transforms.Resize(256)
        ])

    train_dataset = torchvision.datasets.ImageFolder(root='./data/train/', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset)
    test_dataset = torchvision.datasets.ImageFolder(root='./data/test/', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset)

    """Device to train"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """Display datasets (for jupyter-notebook)"""
    # show_batch(train_loader)
    # show_batch(test_loader)

    """Model instance, loss function and optimizer"""
    model = CNN.CNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    """Main training steps"""
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            # Backpropagation and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

    """Accuracy check"""
    model_evaluate(model, test_loader)

    """Single image evaluation"""
    model_evaluate_single(model, './data/predict/img.jpg')
