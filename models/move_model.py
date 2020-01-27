import torch
import torch.nn as nn
import torch.nn.functional as F


class MOVEModel(nn.Module):
    """
    Model object for MOVE.
    The explanation of the design choices can be found at https://arxiv.org/abs/1910.12551.
    """

    def __init__(self, emb_size=16000, sum_method=4, final_activation=3):
        """
        Initializing the network
        :param emb_size: the size of the final embeddings produced by the model
        :param sum_method: the summarization method for the model
        :param final_activation: final activation to use for the model
        """
        super().__init__()

        self.prelu1 = nn.PReLU(init=0.01)
        self.prelu2 = nn.PReLU(init=0.01)
        self.prelu3 = nn.PReLU(init=0.01)
        self.prelu4 = nn.PReLU(init=0.01)
        self.prelu5 = nn.PReLU(init=0.01)

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=256,
                               kernel_size=(12, 180),
                               bias=True)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.key_pool = nn.MaxPool2d(kernel_size=(12, 1))

        self.conv2 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               bias=True)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               dilation=(1, 20),
                               bias=True)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               bias=True)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        if sum_method in [0, 1, 2]:
            self.conv5 = nn.Conv2d(in_channels=256,
                                   out_channels=256,
                                   kernel_size=(1, 5),
                                   dilation=(1, 13),
                                   bias=True)
        else:
            self.conv5 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(1, 5),
                                   dilation=(1, 13),
                                   bias=True)
        nn.init.kaiming_normal_(self.conv5.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.fin_emb_size = emb_size

        self.autopool_p = nn.Parameter(torch.tensor(0.).float())
        self.sum_method = sum_method
        self.final_activation = final_activation

        lin_bias = True
        if self.final_activation == 3:
            self.lin_bn = nn.BatchNorm1d(emb_size, affine=False)
            lin_bias = False

        self.lin1 = nn.Linear(in_features=256, out_features=emb_size, bias=lin_bias)

    def forward(self, data):
        """
        Defining a forward pass of the network
        :param data: input tensor for the network
        :return: output tensor
        """
        x = self.prelu1(self.conv1(data))

        x = self.key_pool(x)

        x = self.prelu2(self.conv2(x))

        x = self.prelu3(self.conv3(x))

        x = self.prelu4(self.conv4(x))

        x = self.prelu5(self.conv5(x))

        if self.sum_method == 0:
            x = torch.max(x, dim=3, keepdim=True).values
        elif self.sum_method == 1:
            x = torch.mean(x, dim=3, keepdim=True)
        elif self.sum_method == 2:
            weights = self.autopool_weights(x)
            x = torch.sum(x * weights, dim=3, keepdim=True)
        elif self.sum_method == 3:
            x = torch.sum(x[:, :256] * torch.nn.functional.softmax(x[:, 256:], dim=3), dim=3, keepdim=True)
        else:
            weights = self.autopool_weights(x[:, :256])
            x = torch.sum(x[:, 256:] * weights, dim=3, keepdim=True)

        x = x.view(-1, 256)
        x = self.lin1(x)

        if self.final_activation == 1:
            x = torch.sigmoid(x)
        elif self.final_activation == 2:
            x = torch.tanh(x)
        elif self.final_activation == 3:
            x = self.lin_bn(x)
        else:
            x = x

        return x

    def autopool_weights(self, data):
        """
        Calculating the autopool weights for a given tensor
        :param data: tensor for calculating the softmax weights with autopool
        :return: softmax weights with autopool
        """
        x = data * self.autopool_p
        max_values = torch.max(x, dim=3, keepdim=True).values
        softmax = torch.exp(x - max_values)
        weights = softmax / torch.sum(softmax, dim=3, keepdim=True)

        return weights
