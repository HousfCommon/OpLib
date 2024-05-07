#--just practice--
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from nn.GoldenModelOfLab.TestCase import *
import collections
import numpy as np


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, weight, bias):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        print(input.dtype, self.weight.t().dtype)
        output = input.matmul(self.weight.t())
        output += self.bias
        return output


class Conv2DLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 weight,
                 bias,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1) -> None:
        super(Conv2DLayer, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def max_pool2d(input, kernel_size, stride, padding):
    # print("input", input.dtype)
    input = input.numpy()
    N, C, H, W = input.shape
    pool_w, pool_h = kernel_size
    stride_w, stride_h = stride
    out_h = int(1 + (H - pool_h) / stride_h)
    out_w = int(1 + (W - pool_w) / stride_w)
    # print("H, W" ,out_h, out_w)
    # print("input", input.shape)
    # print("pool,w,h", pool_h, pool_w)

    col = im2col(input, pool_h, pool_w, stride, padding)
    # print("col_bef", col)
    col = col.astype(np.float32)
    # print("col", col.dtype)
    col = col.reshape(-1, pool_h * pool_w)
    # print("col2", col.shape)

    # arg_max = np.argmax(col, axis=1)
    out = np.max(col, axis=1)
    # print("out_trans", out)

    out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
    # print("out", out)

    return torch.from_numpy(out)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    stride_w, stride_h = stride
    out_h = (H + 2*pad - filter_h)//stride_h + 1
    out_w = (W + 2*pad - filter_w)//stride_w + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # print("img" ,img.shape)
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # print("col", col.shape)

    for y in range(filter_h):
        y_max = y + stride_h*out_h
        for x in range(filter_w):
            x_max = x + stride_w*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride_h, x:x_max:stride_w]
    # print("col_af", col.shape)

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    # print("col_aft", col)
    return col


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps, weight, bias):
        super(LayerNorm, self).__init__()
        normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


def main():
    input = torch.randn(3, 4, 5) # b,n,l
    weight = torch.randn(5)
    bias = torch.randn(5)
    print(input, weight, bias)
    # print(input.shape[-1])
    LN = LayerNorm(input.shape[-1], 1e-5, weight, bias)
    print(LN(input))

    LN2 = nn.LayerNorm(input.shape[-1], 1e-5)
    param_dict = collections.OrderedDict()
    param_dict['weight'] = weight
    param_dict['bias'] = bias
    LN2.load_state_dict(param_dict)
    print(LN2(input))


    # model = Conv2DTesting(input_channels=3, mid_channels=5, output_channels=2)
    # input_data = torch.randn(1, 3, 10, 11)
    # a_real = model(input_data)
    #
    # param_list = collections.OrderedDict()
    # for name, param in model.named_parameters():
    #     param_list[name] = param
    #     print(name ,param.shape)
    #     print('------------------')
    #
    # weight_file0 = open('/Users/huanghuangtao/Desktop/weight_0.txt', 'w')
    # # weight_file1 = open('/Users/huanghuangtao/Desktop/weight_1.txt', 'w')
    #
    # Conv0 = Conv2DLayer(in_channels=3, out_channels=5, kernel_size=5, padding=2, weight=param_list['conv1.weight'], bias=param_list['conv1.bias'])
    # Conv1 = Conv2DLayer(in_channels=5, out_channels=2, kernel_size=5, weight=param_list['conv2.weight'], bias= param_list['conv2.bias'])
    # a_ = Conv0(input_data)
    # a = Conv1(a_)
    #
    # print(a_real.shape, file=weight_file0)
    # print(a_real, file=weight_file0)
    # print('-------------------', file=weight_file0)
    # print(a.shape, file=weight_file0)
    # print(a, file=weight_file0)


if __name__ == '__main__':
    main()