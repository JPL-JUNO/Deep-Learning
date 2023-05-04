"""
@Description: Deep Learning Development with PyTorch
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-03 10:55:00
"""
import torch
from torchvision.datasets import CIFAR10

train_data = CIFAR10(root='./train/',
                     train=True,
                     download=True)
print(train_data)
print(len(train_data))
print(train_data.data.shape)
print(train_data.targets)
print(train_data.classes)
print(train_data.class_to_idx)

print(type(train_data[0]))
print(len(train_data))
data, label = train_data[0]
print(data)
print(type(label))
print(label)
print(train_data.classes[label])

test_data = CIFAR10(root='./test/',
                    train=False,
                    download=True)
print(test_data)
print(len(test_data))
print(test_data.data.shape)

# Data Transforms
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(.4914, .4822, .4465),
        std=(.2032, .1994, .2010)
    )
])

train_data = CIFAR10(root='./train/',
                     train=True,
                     download=True,
                     transform=train_transforms)
print(train_data)
print(train_data.transforms)

data, label = train_data[0]
print(type(data))
print(data.size())
print(data)


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (.4914, .4822, .4465),
        (.2023, .1994, .2010)
    )
])
test_data = CIFAR10(root='./test/',
                    train=False,
                    transform=test_transform)
print(test_data)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=16, shuffle=True
)

data_batch, labels_batch = next(iter(train_loader))
print(data_batch.size())
print(labels_batch.size())

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=16, shuffle=False)
# these is usually no need to shuffle the test data and like to repeat test results
