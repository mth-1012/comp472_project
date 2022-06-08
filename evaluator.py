from torch.utils.data import DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision


def model_evaluate(model, test_loader):
    print('==== evaluate ====')
    tp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
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
            # Since we know batch_size = 1 for test_loader
            # model says
            pre = predicted[0].item()  # 0, 1, 2, 3
            # what it should be
            lab = labels[0].item()     # 0, 1, 2, 3
            # okay, so
            if pre == lab:
                tp[pre] += 1
                for x in range(0, 4):
                    if x != pre: tn[x] += 1

            else:
                for x in range(0, 4): tn[x] += 1
                tn[pre] -= 1
                tn[lab] -= 1
                fp[pre] += 1
                fn[lab] += 1
        print('Test Accuracy of the model on the {} test images: {} %'
              .format(total, (correct / total) * 100))

    print('==== statistics ====')
    for i in range (0, 4):
        print('Class:', classes[i])
        print('> TP = {}, TN = {}, FN = {}. FP = {}'.format(tp[i], tn[i], fn[i], fp[i]))
        print('> Accuracy = {} %'.format(100*(tp[i]+tn[i])/total))
        print('> Precision = {} %'.format(100*tp[i]/(tp[i]+fp[i])))
        print('> Recall = {} %'.format(100*tp[i]/(tp[i]+fn[i])))
        print('> F1-measure = {} %'.format(100*tp[i]/(tp[i]+(fp[i]+fn[i])/2)))
    print('True Positive:', tp)
    print('True Negative:', tn)
    print('False Negative:', fn)
    print('False Positive:', fp)


def model_evaluate_single(model, image_dir):
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


if __name__ == '__main__':
    print('Evaluator')

    """Transform functions"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(32),
            transforms.CenterCrop(32)
        ])

    test_dataset = torchvision.datasets.ImageFolder(root='./data/test/', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset)
    classes = test_dataset.classes

    """Device to evaluate"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """Restore model"""
    model = model_restore()

    """Accuracy check"""
    model_evaluate(model, test_loader)

    """Single image evaluation"""
    model_evaluate_single(model, './data/predict/img.jpg')
    # model_evaluate_single(model, './data/predict/img101.png')
    # model_evaluate_single(model, './data/predict/img292.png')
