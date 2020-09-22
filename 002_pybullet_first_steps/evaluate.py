#!/usr/bin/env python3

import torch.utils.data
import torchvision

from train import Net, SegmentationDataset, PATH_NETWORK

PATH_EVALUATION_SET = './evaluation_set/'

import matplotlib.pyplot as plt


def show_prediction(image_in, prediction):
    plt.figure()
    plt.tight_layout()

    transform_to_pil = torchvision.transforms.ToPILImage()

    plt.subplot(1, 2, 1)
    plt.imshow(transform_to_pil(image_in))

    plt.subplot(1, 2, 2)
    plt.imshow(transform_to_pil(prediction))

    plt.show()


def evaluate(evalset):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used for training: {}".format(device))

    net = Net().to(device)
    net.load_state_dict(torch.load(PATH_NETWORK))

    image_in = evalset[22]['image_in'].to(device)

    prediction = net(image_in.unsqueeze(0)).squeeze(0)

    print("max: {}, min: {}".format(prediction.max(), prediction.min()))



    show_prediction(image_in.to('cpu'), prediction.to('cpu'))


def main():

    dataset = SegmentationDataset(
        root_dir=PATH_EVALUATION_SET
    )

    evaluate(dataset)


if __name__ == '__main__':
    main()
