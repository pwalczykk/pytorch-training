#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_layer_0 = nn.Linear(in_features=11, out_features=55)
        self.hidden_layer_activation_0 = nn.Tanh()
        self.hidden_layer_1 = nn.Linear(in_features=55, out_features=11)
        self.hidden_layer_activation_1 = nn.Tanh()
        self.output_layer = nn.Linear(in_features=11, out_features=1)

    def forward(self, x):
        x = self.hidden_layer_0(x)
        x = self.hidden_layer_activation_0(x)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_activation_1(x)
        output = self.output_layer(x)
        return output


def load_training_data():
    wineq_path = "./winequality-red.csv"
    wineq_text = np.loadtxt(wineq_path, dtype=np.str, delimiter=";", max_rows=1)
    wineq_np = np.loadtxt(wineq_path, dtype=np.float32, delimiter=";", skiprows=1)
    wineq_tensor = torch.from_numpy(wineq_np)

    sources_tensor = wineq_tensor[:, :-1].float()
    targets_tensor = wineq_tensor[:, -1].float()

    return sources_tensor, targets_tensor


def train_model(sources_tensor, targets_tensor, epochs):
    # type: (torch.Tensor, torch.Tensor, int) -> Net
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        # Train
        model.train()
        for i in range(len(sources_tensor)):
            source = sources_tensor[i]
            target = targets_tensor[i].unsqueeze(0)
            optimizer.zero_grad()
            output = model(source)
            loss = F.smooth_l1_loss(output, target)
            loss.backward()
            optimizer.step()
            # scheduler.step()
        print(f'Train Epoch: {epoch}')
        print(f'Train Loss: {loss.item()}')

        # Test
        print("Predictions:")
        for i in range(10):
            print(f"Target: {targets_tensor[i]} | Prediction: {model(sources_tensor[i])}\n")
        # model.eval()
        # test_loss = 0
        # with torch.no_grad():
        #     for i in range(len(sources_tensor)):
        #         source = sources_tensor[i]
        #         target = targets_tensor[i].unsqueeze(0)
        #         output = model(source)
        #         test_loss += F.l1_loss(output, target)
        #
        #     test_loss /= len(sources_tensor)
        #     print(f'Test Loss: {test_loss}')

    return model


def main():
    sources_tensor, targets_tensor = load_training_data()

    model = train_model(sources_tensor, targets_tensor, epochs=100)

    # torch.save(model.state_dict(), "winequality-net-simple-v1.0.pt")


if __name__ == '__main__':
    main()
