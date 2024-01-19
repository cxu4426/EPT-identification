import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Source(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3,32,5,stride=2,padding=2)
        self.pool = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(32,64,3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(64,16,3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(16,32,3,stride=1,padding=1)
        self.fc_1 = nn.Linear(2048,256)
        self.fc_2 = nn.Linear(256, 3)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

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
        x = self.conv1(x) #32 256 256
        x = F.relu(x)
        # maxpool
        x = self.pool(x) #32 128 128
        # layer 2 - conv
        x = self.conv2(x) #64 128 128
        x = F.relu(x)
        # maxpool
        x = self.pool(x) #64 64 64
        # layer 3 - conv
        x = self.conv3(x) #16 64 64
        x = F.relu(x)
        # maxpool
        x = self.pool(x) #16 32 32
        # layer 4 - conv
        x = self.conv4(x) #32 32 32
        x = F.relu(x)
        # layer 5 - fully connected
        x = torch.flatten(x, 1) #turn array into vector for fc layer
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x
