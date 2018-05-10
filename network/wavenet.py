import torch
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np


class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet"""
    def __init__(self, channels, dilation=1):
        super(DilatedCausalConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(channels, channels,
                                    kernel_size=2, stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=0,  # Fixed for WaveNet dilation
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        return output



class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        """
        Residual block
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :param dilation:
        """
        super(ResidualBlock, self).__init__()

        self.dilated = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = self.dilated(x)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid

        # Residual network
        output = self.conv_res(gated)
        input_cut = x[:, :, -output.size(2):]
        output += input_cut

        # Skip connection
        skip = self.conv_skip(gated)
	#print("[network:ResidualBlock:conv_skip]")
	#print(skip.data.shape)
        skip = skip[:, :, -skip_size:]

        return output, skip


class ResStack(nn.Module):
    def __init__(self, in_out_size, res_size, stack_size, layer_size):
	super(ResStack, self).__init__()

	self.in_out_size = in_out_size
	self.res_size = res_size
	self.stack_size = stack_size
	self.layer_size = layer_size

	self.res_stack = self.buildStack(res_size, in_out_size)
	#self.skip = self._skipConv(res_size, in_out_size) 
	#self.res = self._resConv(res_size, res_size)
	#self.dilated = DilatedCausalConv1d(res_size, dilation=dilation)

    #def _dilatedCasualConv(self, in_size, out_size, dilation):
    #    return nn.Conv1d(in_size, out_size, kernel_size=2, stride=1, padding=1, dilation=dilation)

    #def _skipConv(self, in_size, out_size):
	#return nn.Conv1d(in_size, out_size, kernel_size=1)

    #def _resConv(self, in_size, out_size):
	#return nn.Conv1d(in_size, out_size, kernel_size=1)

    def _residualBlk(self, res_size, skip_size, dilation):
	block = ResidualBlock(res_size, skip_size, dilation)

	if torch.cuda.device_count() >1:
	    block = torch.nn.DataParallel(block)

	if torch.cuda.is_available():
            block.cuda()

        return block

    def buildStack(self, res_size, skip_size):
        res_blocks = []
        for s in range(0,self.stack_size):
            for l in range(0,self.layer_size):
		block = self._residualBlk(self.res_size, self.in_out_size, 2**l)
		#if torch.cuda.is_available():
		#    block.cuda()
                res_blocks.append(block)
        return res_blocks 

    def forward(self, x, output_size):
	skip_connections = []

	for res_block in self.res_stack:
	    x, skip = res_block(x, output_size)# self.in_out_size)
	    skip_connections.append(skip)

	return torch.stack(skip_connections)


class LinearNet(nn.Module):
    def __init__(self, in_size, out_size):
	super(LinearNet, self).__init__()

	self.conv1 = nn.Conv1d(in_size, out_size, kernel_size=1)
	self.conv2 = nn.Conv1d(in_size, out_size, kernel_size=1)

	self.relu = nn.ReLU()
	self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
	x = self.relu(x)
	x = self.conv1(x)
	x = self.relu(x)
	x = self.conv2(x)

	return self.softmax(x)#nn.Softmax(x, dim=1)
	#return nn.Softmax(x[1])


class WaveNet(nn.Module):
    def __init__(self, in_out_size=256, res_size=512, stack_size=5, layer_size=10):
        super(WaveNet, self).__init__()

	self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)
	#self.causalConv = self._causalConv(in_out_size, res_size)
	self.causalConv = self._causalConv(in_out_size,res_size)
	self.resStack = ResStack(in_out_size, res_size, stack_size, layer_size)
	self.linear = LinearNet(in_out_size, in_out_size)


    def _causalConv(self, in_size, out_size):
	return nn.Conv1d(in_size, out_size, kernel_size=2, stride=1, padding=1, bias=False)


    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = np.sum(layers)

        return int(num_receptive_fields)


    def calc_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_fields

        self.check_input_size(x, output_size)

        return output_size

    def check_input_size(self, x, output_size):
        if output_size < 1:
	    message = 'Input size has to be larger than receptive_fields\n'
	    message += 'Input size: {0}, Receptive fields size: {1}, Output size: {2}'.format(int(x.size(2)), self.receptive_fields, output_size)

	    print(message)
            #raise InputSizeError(int(x.size(2)), self.receptive_fields, output_size)


    def forward(self, x):
	#print('[network:input]', x, x.data.shape)
        x = x.transpose(1, 2)
        #print('[network:transpose]', x, x.data.shape)

	#for last output, check output size to prevent error
	output_size = self.calc_output_size(x)

	x = self.causalConv(x)
	#print('[network:causal]', x, x.data.shape)
	x = x[:, :, :-1]#remove the last value to keep causal
	#print('[network:remove the last value to keep causal]', x, x.data.shape)
	x = self.resStack(x, output_size)
	#print('[network:residual satck]', x, x.data.shape)
	x = torch.sum(x, dim=0)
	#print('[network:sum skip connections]', x, x.data.shape)
	x = self.linear(x)
	#print('[network:linear]', x, x.data.shape)

	x = x.transpose(1,2).contiguous()
	#print('[network:transpose & contiguous]', x, x.data.shape)

        return x











