#--just practice--
import os, sys
import torch
import torch.nn as nn
import collections
import nn.GoldenModelOfLab.quanti as qt
import nn.GoldenModelOfLab.Layer as L
import torch.nn.functional as F


class GoldenSpatialModel(nn.Module):
    def __init__(self, in_channels, out_num, weight_lib, batch_size):
        super(GoldenSpatialModel, self).__init__()
        self.batch_size = batch_size
        self.Conv1 = L.Conv2DLayer(in_channels=in_channels, out_channels=64, kernel_size=5, padding=2, weight=weight_lib['conv1.weight'], bias=weight_lib['conv1.bias'])
        self.Conv2 = L.Conv2DLayer(in_channels=64, out_channels=128, kernel_size=5, weight=weight_lib['conv2.weight'], bias=weight_lib['conv2.bias'])
        self.fc3 = L.LinearLayer(1280, out_num, weight=weight_lib['fc3.weight'], bias=weight_lib['fc3.bias'])

    def forward(self, input_data):
        x1 = F.relu(self.Conv1(input_data))
        print("x1", x1.dtype)
        x1_ = L.max_pool2d(x1, (2,2), (2,2), 0)
        print("x1_", x1_.dtype)
        x1_scale = qt.data_scale_gen(x1_, 'spatial_conv2', 'conv2_data')
        x1_ = qt.quantize_data(x1_, x1_scale)
        x2 = F.relu(self.Conv2(x1_))
        print("x2", x2.dtype)
        x2_ = L.max_pool2d(x2, (2,2), (2,2), 0)
        print("x2_", x2_.shape)

        y = x2_.reshape(self.batch_size, -1)
        print("y", y.shape, self.batch_size)
        y_pred = self.fc3(y)

        return y_pred

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features


def main():
    param_list = collections.OrderedDict()
    param_list['conv1.weight'] = torch.randn(5, 3, 5, 5)
    param_list['conv1.bias'] = torch.randn(5)
    param_list['Linear1.weight'] = torch.ones(2, 10)
    param_list['Linear1.bias'] = torch.randn(2)
    for key in ['conv1.weight', 'conv1.bias', 'Linear1.weight', 'Linear1.bias']:
        scale = qt.weight_scale_gen(param_list[key], key, 'Linear_input')
        param_list[key] = qt.quantize_data(param_list[key], scale)

    with torch.no_grad():
        model = GoldenSpatialModel(5, 1053, param_list)
        # for name, param in model.named_parameters():
        #     param.requires_grad_(False)
        #     param.int()
        print("another test:")
        #     print(model.named_parameters()['Linear.weight'])

        #     print(param_list)
        # model.load_state_dict(param_list)
        # print(list(model.named_parameters()))
        print('----------------------')
        # print(model.Linear0.weight.dtype)
        input_data = torch.randn(1, 3, 10, 11)
        x_scale = qt.data_scale_gen(input_data, 'Linear0', 'Linear_input')
        input_data = qt.quantize_data(input_data, x_scale)
        res = model(input_data)
        print(res.shape)


if __name__ == '__main__':
    main()