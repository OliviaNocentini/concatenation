from FashionCNN import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

if __name__ == '__main__':

    batch_size = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Load the training and test dataset")
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
    transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
    transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size)

    # Examine a sample
    dataiter = iter(train_loader)

    images, labels = dataiter.next()
    # print("images.shape",images.shape)
    # plt.imshow(images[0].numpy().squeeze(), cmap = 'Greys_r')

    # loading CNN
    model = FashionCNN()
    model.to(device)

    # defining error
    error = nn.CrossEntropyLoss()

    # defining lr and optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 2
    count = 0

    # Lists for visualization of loss and accuracy 
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []


    print("Back propagation starting")
    for epoch in range(num_epochs):
        print("Epoch ", epoch)
        running_loss = 0.0

        for images, labels in train_loader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            # visualize = false because nto interested to visualize the hog image
            fd = np.array([hog(image.numpy().squeeze(), orientations=8, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=False, multichannel=False) for image in images])
            """
            Uncomment me if you want to check the hog of the first figure 
            fd, hog_image = hog(images[0].numpy().squeeze(), orientations=8, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True, multichannel=False)

            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.axis('off')
            ax1.imshow(images[0].numpy().squeeze(), cmap = 'Greys_r')
            ax1.set_title('Input image')

            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()
            break
            """

            train = Variable(images.view(batch_size, 1, 28, 28))
            fd = fd.astype(np.float32)
            data_vector = torch.from_numpy(fd)
            labels = Variable(labels)

            # Forward pass 
            outputs = model(train, data_vector)
            loss = error(outputs, labels)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            # Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            count += 1
            running_loss += loss.item()

            print("count ", count)

            if count % 10:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, count + 1, running_loss / 2000))
                running_loss = 0.0

            # Testing the model
            if not (count % 50):  # It's same as "if count % 50 == 0"
                total = 0
                correct = 0

                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)

                    test = Variable(images.view(batch_size, 1, 28, 28))

                    outputs = model(test, data_vector)

                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 500):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
