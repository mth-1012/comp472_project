from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import CNN as CNN
from old import evaluator as evaluator

# Transform functions
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize(32),
        transforms.CenterCrop(32)
    ])


def model_save(model):
    torch.save(model, 'model.pt')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('COMP 472 Project')
    print('AI Face Mask Detector')

    """Load dataset"""
    num_epochs = 10
    num_classes = 4
    learning_rate = 0.001

    # Dataset
    train_dataset = torchvision.datasets.ImageFolder(root='../data/train/', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=6)
    test_dataset = torchvision.datasets.ImageFolder(root='../data/test/', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset)
    classes = train_dataset.classes

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
    print('==== training ====')
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

    """Save model"""
    model_save(model)

    """Evaluate"""
    # evaluator.evaluate(test_dataset)
