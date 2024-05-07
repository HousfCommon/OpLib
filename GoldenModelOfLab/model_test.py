#--just practice--
import numpy as np
import torch
import collections
from nn.GoldenModelOfLab.SpacialModel import GoldenSpatialModel
import nn.GoldenModelOfLab.quanti as qt
from nn.LabOfNPmems.spatialmodel import SpatialModel


def main():
    param_list = load_spacial_weight()
    checkpoint = torch.load('/Users/huanghuangtao/Desktop/check_point_50', map_location='cpu')
    batch_size = 10
    with torch.no_grad():
        model0 = GoldenSpatialModel(5, 1053, param_list, batch_size=batch_size)
        model1 = SpatialModel(num_input_channels=5, out_num=1053, dropout_p=0.1)
        model1.eval()
        model1.load_state_dict(checkpoint['model_spatial'])
        # for name, param in model.named_parameters():
        #     param.requires_grad_(False)
        #     param.int()
        print("another test:")

        print('----------------------')
        # print(model.Linear0.weight.dtype)
        input_data = torch.randn(batch_size, 5, 17, 31)

        # x_scale = qt.data_scale_gen(input_data, 'Linear0', 'Linear_input')
        # input_data = qt.quantize_data(input_data, x_scale)
        res = model0(input_data)
        ret = model1(input_data)
        print(res.shape, ret.shape)

        file_res = open('/Users/huanghuangtao/Desktop/weight_upgd.txt', 'w')
        file_res1 = open('/Users/huanghuangtao/Desktop/weight_upgd1.txt', 'w')
        print(res.numpy(), file=file_res)
        print(ret.numpy(), file=file_res1)
        p = (ret-res)/ret
        p = p.numpy()
        p = np.sum(np.abs(p))/(1053*batch_size)
        print("result:", p)

        file_res.close()
        file_res1.close()


def load_spacial_weight():
    # map_location = torch.device('cpu')
    np.set_printoptions(threshold=np.inf)

    checkpoint = torch.load('/Users/huanghuangtao/Desktop/check_point_50', map_location='cpu')
    model_spat = checkpoint['model_spatial']

    # for key in model_spat.keys():
    #     print(key)

    # load data of conv1
    conv1_bn1_weight = model_spat['conv_bn1.weight']
    conv1_bn1_bias = model_spat['conv_bn1.bias']
    conv1_bn1_running_mean = model_spat['conv_bn1.running_mean']
    conv1_bn1_running_var = model_spat['conv_bn1.running_var']
    # print(conv1_bn1_weight.shape, conv1_bn1_bias.shape, conv1_bn1_running_mean.shape, conv1_bn1_running_var.shape)
    conv1_bias = model_spat['conv1.bias']
    conv1_weight = model_spat['conv1.weight']
    # print(conv1_bias.shape, conv1_weight.shape)

    # gen conv1_weight
    sqrt_bn1_var = torch.sqrt(conv1_bn1_running_var)
    # print(conv1_weight.shape, conv1_bn1_weight.shape)
    bn1_weight = conv1_bn1_weight.reshape(-1, 1, 1, 1)
    bn1_var = sqrt_bn1_var.reshape(-1, 1, 1, 1)
    bn1 = bn1_weight / bn1_var
    conv1_weight_upgd = conv1_weight * bn1_weight / bn1_var

    # print(bn1_weight[1], bn1_var[1])
    # print(conv1_weight[1][1], bn1[1], conv1_weight_upgd[1][1], conv1_weight_upgd.shape)
    # print('_____________')

    # gen conv1_bias
    conv1_bias_1 = conv1_bn1_weight / sqrt_bn1_var * (conv1_bias - conv1_bn1_running_mean)
    conv1_bias_upgd = conv1_bn1_bias + conv1_bias_1

    # print(conv1_bias_1[1], conv1_bias[1], conv1_bias_upgd[1])
    # print('_____________')

    # load data of conv2
    conv2_bn2_weight = model_spat['conv_bn2.weight']
    conv2_bn2_bias = model_spat['conv_bn2.bias']
    conv2_bn2_running_mean = model_spat['conv_bn2.running_mean']
    conv2_bn2_running_var = model_spat['conv_bn2.running_var']
    # print(conv2_bn2_weight.shape, conv2_bn2_bias.shape, conv2_bn2_running_mean.shape, conv2_bn2_running_var.shape)
    conv2_bias = model_spat['conv2.bias']
    conv2_weight = model_spat['conv2.weight']
    # print(conv2_bias.shape, conv2_weight.shape)

    # gen conv2_weight
    sqrt_bn2_var = torch.sqrt(conv2_bn2_running_var)
    # print(conv2_weight.shape, conv2_bn2_weight.shape)
    bn2_weight = conv2_bn2_weight.reshape(-1, 1, 1, 1)
    bn2_var = sqrt_bn2_var.reshape(-1, 1, 1, 1)
    bn2 = bn2_weight / bn2_var
    conv2_weight_upgd = conv2_weight * bn2_weight / bn2_var

    # print(bn2_weight[1], bn2_var[1])
    # print(conv2_weight[1][1], bn2[1], conv2_weight_upgd[1][1], conv2_weight_upgd.shape)
    # print('_____________')

    # gen conv2_bias
    conv2_bias_1 = conv2_bn2_weight / sqrt_bn2_var * (conv2_bias - conv2_bn2_running_mean)
    conv2_bias_upgd = conv2_bn2_bias + conv2_bias_1

    # print(conv2_bias_1[1], conv2_bias[1], conv2_bias_upgd[1])
    # print('_____________')

    # load data of fc3
    fc3_weight = model_spat['fc3.weight']
    fc3_bias = model_spat['fc3.bias']

    # print(fc3_weight.shape, fc3_bias.shape)

    param_list = collections.OrderedDict()
    param_list['conv1.weight'] = conv1_weight_upgd
    param_list['conv1.bias'] = conv1_bias_upgd
    param_list['conv2.weight'] = conv2_weight_upgd
    param_list['conv2.bias'] = conv2_bias_upgd
    param_list['fc3.weight'] = fc3_weight
    param_list['fc3.bias'] = fc3_bias

    return param_list


if __name__ == '__main__':
    main()