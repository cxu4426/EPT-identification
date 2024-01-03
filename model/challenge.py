"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Challenge(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        ## TODO: define your model architecture
        self.conv1 = nn.Conv2d(3,16,5,stride=2,padding=2)
        self.conv2 = nn.Conv2d(16,64,5,stride=2,padding=2)
        self.conv3 = nn.Conv2d(64,8,5,stride=2,padding=2)
        self.pool = nn.MaxPool2d(2,stride=2)
        self.dropout1 = nn.Dropout()
        self.fc_1 = nn.Linear(32,8)
        self.fc_2 = nn.Linear(32,2)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        ## TODO: initialize the parameters for your network
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        for fc in [self.fc_1, self.fc_2]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """
        N, C, H, W = x.shape
        ## TODO: implement forward pass for your network
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # x = self.dropout1(x) # source train only

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x = self.dropout1(x) # source train only

        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        # x = self.fc_1(x) # source train
        x = self.fc_2(x) # target train

        return x
