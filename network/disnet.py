import torch
import torch.nn as nn
import torch.nn.functional as F



class Discriminator(nn.Module):
    def __init__(self, input_size=256, output_size=1):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=2, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=2, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=2, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv1d(512, output_size, kernel_size=2, stride=1)#, padding=1)
        

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.leaky_relu(self.conv1(x))
        #print('x_conv1: ',x)
        x = x[:, :, :-1]
        x = F.leaky_relu(self.conv2(x))
        #print('x_conv2: ',x)
        x = x[:, :, :-1]
        x = F.leaky_relu(self.conv3(x))
        #print('x_conv3: ',x)
        x = x[:, :, :-1]
        x = F.leaky_relu(self.conv4(x))
        #print('x_conv4: ',x)
        #x = x[:, :, :-1]
        return F.sigmoid(self.conv5(x))


