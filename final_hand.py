#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  torchvision,torch
import  torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import sys

'''
def imshow(img):
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(npimg.transpose((1,2,0)))
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# In[2]:
if __name__ == '__main__':
    print(sys.argv[1])
    trainset = torchvision.datasets.MNIST(root='./mnist/',train=True, download=True, transform= transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=4,shuffle=True, num_workers=4)

    testset = torchvision.datasets.MNIST(root='./mnist/',train=False,download=True, transform= transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=4,shuffle=True, num_workers=4)


    # In[3]:



    classes = ('0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9')
    print('01',trainloader,'\n')
    dataiter = iter(trainloader)
    print('02',dataiter)
    images, labels = dataiter.next()
    print(images.shape,'\n')
    print(labels)


        
    #imshow(torchvision.utils.make_grid(images))


    # In[5]:





    device = torch.device("cuda:0" if torch.cuda.is_available() and sys.argv[1]=="GPU" else "cpu")
    #device = torch.device("cpu")

    print(device)
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    # In[6]:




    start = time.time()
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
        # print(outputs.shape,'\n')
        # print(labels.shape,'\n')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

                
    end = time.time()
    print('Finished Training')
    print('training time is %6.3f' % (end - start))


    # In[10]:


    dataiter = iter(testloader)
    images, labels = dataiter.next()

    #imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted:   ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))


    # In[12]:


    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


    # In[ ]:




