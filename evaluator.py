from torch.utils.data import DataLoader
from PIL import Image
import numpy
import torch
import torchvision.transforms as transforms
import torchvision

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(32),
            transforms.CenterCrop(32)
        ])


def model_evaluate(model, test_set, device):
    test_loader = torch.utils.data.DataLoader(test_set)
    classes = test_set.classes
    print('==== evaluate ====')
    cm = numpy.zeros((4, 4))
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
            print('Highest chance to be: ', classes[predicted[0].item()])
            pre = predicted[0].item()   # 0, 1, 2, 3
            lab = labels[0].item()      # 0, 1, 2, 3
            cm[pre][lab] += 1
        print('Test Accuracy of the model on the {} test images: {} %'
              .format(total, (correct / total) * 100))

    print('==== statistics ====')
    [print(cm[i]) for i in range(0, 4)]


def model_evaluate_single(model, image_dir, transform, classes, device):
    print('==== single evaluate ====')
    single = Image.open(image_dir).convert('RGB')
    input = transform(single)
    input = input.unsqueeze(0)
    input = input.to(device)
    output = model(input)
    _, predicted = torch.max(output.data, 1)
    print('Highest chance to be: ', classes[predicted[0].item()])


def model_restore():
    return torch.load('model.pt')


def evaluate(test_set):
    print('Evaluator')

    """Device to evaluate"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """Restore model"""
    model = model_restore()

    """Accuracy check"""
    model_evaluate(model, test_set, device)

    """Single image evaluation"""
    model_evaluate_single(model, './data/predict/img.jpg', transform, classes=test_set.classes, device=device)
    # model_evaluate_single(model, './data/predict/img101.png')
    # model_evaluate_single(model, './data/predict/img292.png')


if __name__ == '__main__':
    # Dataset for testing only
    test_dataset = torchvision.datasets.ImageFolder(root='./data/test/', transform=transform)
    evaluate(test_dataset)
