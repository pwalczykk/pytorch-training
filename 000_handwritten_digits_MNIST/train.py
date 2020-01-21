#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


class Network():

    def __init__(self):
        self.trainloader = None # type: DataLoader
        self.model = None  # type: nn.Sequential

    def load_data(self):
        # Define a transform to normalize the data
        transform = transforms.Compose(
            transforms=[
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )

        # Download and load the training data
        trainset = datasets.MNIST(
            root='~/.pytorch/MINST_data',
            download=False,
            train=True,
            transform=transform
        )

        self.trainloader = DataLoader(
            dataset=trainset,
            batch_size=1,
            shuffle=True
        )

    def train(self, epochs):

        # model = Network()
        self.model = nn.Sequential(nn.Linear(784, 128),
                              nn.ReLU(),
                              nn.Linear(128, 64),
                              nn.ReLU(),
                              nn.Linear(64, 10),
                              nn.LogSoftmax(dim=1))

        criterion = nn.NLLLoss()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.003)

        for e in range(epochs):
            running_loss = 0
            dataset_id = -1
            for images, labels in self.trainloader:
                dataset_id += 1
                if dataset_id < 40000:
                    # Flatten MINST images into a 784 long vector
                    images = images.view(images.shape[0], -1)

                    # Training pass
                    optimizer.zero_grad()

                    output = self.model(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                else:
                    pass
            else:
                print(f"Training loss: {running_loss/len(self.trainloader)}")

            self.predict_all()

        print("Training done")

    def predict_all(self):

        bad_predictions = 0

        dataset_id = -1
        for images, labels in self.trainloader:
            dataset_id += 1
            if dataset_id < 40000:
                pass
            else:
                images = images.view(images.shape[0], -1)
                outputs = self.model(images)

                for id in range(len(labels)):
                    output_np = outputs[id].detach().numpy()
                    prediction = output_np.argmax()
                    if prediction != labels[id]:
                        bad_predictions += 1

        print(f"Bad predictions: {bad_predictions}/{len(self.trainloader.dataset)-40000} | "
              f"Percentage: {100.0*bad_predictions/(len(self.trainloader.dataset)-40000)}%")
        print("=============================")

    def predict(self, id):
        output = self.model(self.trainloader.dataset[id][0].view(torch.Size([1, 784]), -1))
        label = self.trainloader.dataset[id][1]

        output_np = output.detach().numpy()
        prediction = output_np.argmax()

        print(f"Prediction: {prediction} | Label: {label}")

        return prediction, label


if __name__ == '__main__':

    network = Network()

    network.load_data()
    network.train(epochs=500)
