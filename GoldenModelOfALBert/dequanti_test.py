#--just practice--
import torch
import nn.GoldenModelOfALBert.Layer as L
from nn.GoldenModelOfALBert.quanti import *
import torch.nn as nn
import collections


class LNetworkConfig():
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


class LNetwork(nn.Module):
    def __init__(self, config, param_list):
        super(LNetwork, self).__init__()
        self.query = L.LinearLayer(config.hidden_size, config.output_size,
                                   param_list['query.weight'],
                                   param_list['query.bias'])
        self.key = L.LinearLayer(config.hidden_size, config.output_size,
                                 param_list['key.weight'],
                                 param_list['key.bias'])

    def forward(self, input):
        query_layer = self.query(input)
        key_layer = self.key(input)
        score = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        return score


def main():
    input_size = 10
    hidden_size = 10
    output_size = 20
    config = LNetworkConfig(input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=output_size)

    param_list = collections.OrderedDict()
    param_list_q = collections.OrderedDict()

    input_data = torch.randn(input_size, hidden_size)
    param_list['query.weight'] = torch.randn(output_size, hidden_size)
    param_list['query.bias'] = torch.randn(output_size)
    param_list['key.weight'] = torch.randn(output_size, hidden_size)
    param_list['key.bias'] = torch.randn(output_size)
    #
    # get scale
    x_scale = data_scale_gen(input_data, 'Linear', 'Linear_input')
    w_scale_0 = weight_scale_gen(param_list['query.weight'], 'Linear0', 'Linear_weight')
    b_scale_0 = bias_scale_gen(param_list['query.bias'], 'Linear0', 'Linear_bias')
    w_scale_1 = weight_scale_gen(param_list['key.weight'], 'Linear1', 'Linear_weight')
    b_scale_1 = bias_scale_gen(param_list['key.bias'], 'Linear1', 'Linear_bias')

    # quantize data
    input_data_q = quantize_data(input_data, x_scale)
    param_list_q['query.weight'] = quantize_data(param_list['query.weight'], w_scale_0)
    param_list_q['query.bias'] = quantize_bias(param_list['query.bias'], b_scale_0)
    param_list_q['key.weight'] = quantize_data(param_list['key.weight'], w_scale_1)
    param_list_q['key.bias'] = quantize_bias(param_list['key.bias'], b_scale_1)

    # initialize network
    # LN用初始数据，LN_q用量化数据
    LN = LNetwork(config=config, param_list=param_list)
    LN_q = LNetwork(config=config, param_list=param_list_q)

    # forward propagation
    output = LN(input_data)
    output_q = LN_q(input_data_q)

    print("real output", output)
    print("quantize output", output_q)


if __name__ == '__main__':
    main()