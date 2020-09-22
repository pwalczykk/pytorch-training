#!/usr/bin/env python3

import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import subprocess

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt

PATH_NETWORK = './segmentation_net.pth'
PATH_TRAINING_SET = 'training_set/'
EPOCHS = 10


# Define network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 60, 5)
        self.conv2 = nn.Conv2d(60, 160, 5)
        self.conv3 = nn.Conv2d(160, 640, 5)
        self.conv4 = nn.Conv2d(640, 160, 5)
        self.conv5 = nn.Conv2d(160, 60, 5)
        self.conv6 = nn.Conv2d(60, 1, 5)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = F.relu(x)

        return x


class SegmentationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform_in = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.transform_out = torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop([216, 296]),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(0.5, 0.5),
            ]
        )

    def __len__(self):
        return int(subprocess.check_output("ls -l " + self.root_dir + "| grep cam0-seg | wc -l", shell=True))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_in = Image.open(self.root_dir + str(idx) + "-cam0-rgb.png")
        image_out = Image.open(self.root_dir + str(idx) + "-cam0-seg.png")

        sample = {'image_in': self.transform_in(image_in), 'image_out': self.transform_out(image_out)}

        return sample


def show_datapoint(datapoint):

    plt.figure()
    plt.tight_layout()

    transform_to_pil = torchvision.transforms.ToPILImage()

    plt.subplot(1, 2, 1)
    plt.imshow(transform_to_pil(datapoint['image_in']))

    plt.subplot(1, 2, 2)
    plt.imshow(transform_to_pil(datapoint['image_out']))

    plt.show()

def train(trainset):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used for training: {}".format(device))

    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True)

    net = Net().to(device)

    # Define loss function

    import torch.optim as optim

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

    # Perform training
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image_in'].to(device), data['image_out'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d/%d, %5d] loss: %.3f' %
                      (epoch + 1, EPOCHS, i + 1, running_loss / 20))
                running_loss = 0.0

    # Save model
    torch.save(net.state_dict(), PATH_NETWORK)

    print('Finished Training')


def main():

    dataset = SegmentationDataset(
        root_dir=PATH_TRAINING_SET)

    train(dataset)


if __name__ == '__main__':
    main()
