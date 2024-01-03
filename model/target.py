"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2
Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import target
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Target(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        ## TODO: define each layer
        self.conv1 = nn.Conv2d(3,16,5,stride=2,padding=2)
        self.pool = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(16,64,5,stride=2,padding=2)
        self.conv3 = nn.Conv2d(64,8,5,stride=2,padding=2)
        self.fc_1 = nn.Linear(32,2)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc_1]
        C_in = self.fc_1.weight.size(1)
        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(self.fc_1.bias, 0.0)

    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """
        N, C, H, W = x.shape

        ## TODO: forward pass
        # layer 1 - conv
        x = self.conv1(x) #[32, 16, 32, 32]
        x = F.relu(x)
        # layer 2 - pool
        x = self.pool(x) #[32, 16, 16, 16]
        # layer 3 - conv
        x = self.conv2(x) #[32, 64, 8, 8]
        x = F.relu(x)
        # layer 4 - pool
        x = self.pool(x) #[32, 64, 4, 4]
        # layer 5 - conv
        x = self.conv3(x) #[32, 8, 2, 2]
        x = F.relu(x)
        # layer 6 - fully connected
        x = torch.flatten(x, 1) #turn array into vector for fc layer
        x = self.fc_1(x)

        return x
