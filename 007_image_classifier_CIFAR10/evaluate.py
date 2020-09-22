#!/usr/bin/env python3

import torch.utils.data
import torchvision

from train import Net, PATH, TRANSFORM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device used for training: {}".format(device))

net = Net().to(device)
net.load_state_dict(torch.load(PATH))

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=TRANSFORM)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
r
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

