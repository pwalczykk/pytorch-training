#!/usr/bin/env python3

# https://blog.tensorpad.com/a-comprehensive-intro-to-pytorch/


import torch
import torch.nn as nn
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class RegNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(11, 64)
        self.l2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.sigmoid(x)
        x = self.l2(x)
        return x


def load_training_data():
    df_red = pd.read_csv('wine-quality/winequality-red.csv', sep=';')
    df_whi = pd.read_csv('wine-quality/winequality-white.csv', sep=';')
    df_red["wine type"] = 'red'
    df_whi["wine type"] = 'white'

    df = pd.concat([df_red, df_whi], axis=0)
    df = df.reset_index(drop=True)

    data = df.loc[:, ~df.columns.isin(['quality', 'wine type'])].values
    target = df.loc[:, df.columns.isin(['quality'])].values

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        target,
        test_size=0.25,
        shuffle=True,
        random_state=42
    )

    return x_train.astype('float32'), x_test.astype('float32'), y_train.astype('float32'), y_test.astype('float32')


def plot_loss(loss_lst):
    plt.plot(loss_lst)
    plt.xlabel('N_Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.01)


def main():

    x_train, x_test, y_train, y_test = load_training_data()

    model = RegNN()
    model.cpu()

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    loss_lst = []
    x_torch = torch.from_numpy(x_train.astype('float32'))
    y_torch = torch.from_numpy(y_train.astype('float32'))

    x_torch_test = torch.from_numpy(x_test.astype('float32'))
    y_torch_test = torch.from_numpy(y_test.astype('float32'))

    for epoch in range(1000):
        optimiser.zero_grad()
        y_pred = model(x_torch)
        loss = criterion(y_pred, y_torch)
        loss_lst.append(loss.data)
        loss.backward()
        optimiser.step()

        print("Epoch: {}    Loss: {}".format(epoch, loss))
        if epoch % 25 == 0:
            plot_loss(loss_lst)

    for i in range(10):
        print("Label: {}    Output: {}".format(y_torch_test[i], model(x_torch_test[i])))

    plt.pause(20)


if __name__ == '__main__':
    main()
