#--just practice--
import numpy as np
import torch
import collections


def main():
    # map_location = torch.device('cpu')
    np.set_printoptions(threshold=np.inf)

    checkpoint = torch.load('/Users/huanghuangtao/Desktop/check_point_50', map_location='cpu')

    for key in checkpoint.keys():
        print(key)

    model_spat = checkpoint['optimizer']
    # print(model_spat)
    for key in model_spat.keys():
        print("sub_", key)

    # print(model_spat['decoder.conv_bn1.weight'])

    # # load data of conv1
    # conv1_bn1_weight = model_spat['decoder.conv_bn1.weight']
    # conv1_bn1_bias = model_spat['decoder.conv_bn1.bias']
    # conv1_bn1_running_mean = model_spat['decoder.conv_bn1.running_mean']
    # conv1_bn1_running_var = model_spat['decoder.conv_bn1.running_var']
    # # print(conv1_bn1_weight.shape, conv1_bn1_bias.shape, conv1_bn1_running_mean.shape, conv1_bn1_running_var.shape)
    # conv1_bias = model_spat['decoder.conv1.bias']
    # conv1_weight = model_spat['decoder.conv1.weight']
    # # print(conv1_bias.shape, conv1_weight.shape)
    #
    # # gen conv1_weight
    # sqrt_bn1_var = torch.sqrt(conv1_bn1_running_var)
    # # print(conv1_weight.shape, conv1_bn1_weight.shape)
    # bn1_weight = conv1_bn1_weight.reshape(-1, 1, 1, 1)
    # bn1_var = sqrt_bn1_var.reshape(-1, 1, 1, 1)
    # bn1 = bn1_weight / bn1_var
    # conv1_weight_upgd = conv1_weight * bn1_weight / bn1_var
    #
    # # print(bn1_weight[1], bn1_var[1])
    # # print(conv1_weight[1][1], bn1[1], conv1_weight_upgd[1][1], conv1_weight_upgd.shape)
    # # print('_____________')
    #
    # # gen conv1_bias
    # conv1_bias_1 = conv1_bn1_weight / sqrt_bn1_var * (conv1_bias - conv1_bn1_running_mean)
    # conv1_bias_upgd = conv1_bn1_bias + conv1_bias_1
    #
    # # print(conv1_bias_1[1], conv1_bias[1], conv1_bias_upgd[1])
    # # print('_____________')
    #
    # # load data of conv2
    # conv2_bn2_weight = model_spat['decoder.conv_bn2.weight']
    # conv2_bn2_bias = model_spat['decoder.conv_bn2.bias']
    # conv2_bn2_running_mean = model_spat['decoder.conv_bn2.running_mean']
    # conv2_bn2_running_var = model_spat['decoder.conv_bn2.running_var']
    # # print(conv2_bn2_weight.shape, conv2_bn2_bias.shape, conv2_bn2_running_mean.shape, conv2_bn2_running_var.shape)
    # conv2_bias = model_spat['decoder.conv2.bias']
    # conv2_weight = model_spat['decoder.conv2.weight']
    # # print(conv2_bias.shape, conv2_weight.shape)
    #
    # # gen conv2_weight
    # sqrt_bn2_var = torch.sqrt(conv2_bn2_running_var)
    # # print(conv2_weight.shape, conv2_bn2_weight.shape)
    # bn2_weight = conv2_bn2_weight.reshape(-1, 1, 1, 1)
    # bn2_var = sqrt_bn2_var.reshape(-1, 1, 1, 1)
    # bn2 = bn2_weight / bn2_var
    # conv2_weight_upgd = conv2_weight * bn2_weight / bn2_var
    #
    # # print(bn2_weight[1], bn2_var[1])
    # # print(conv2_weight[1][1], bn2[1], conv2_weight_upgd[1][1], conv2_weight_upgd.shape)
    # # print('_____________')
    #
    # # gen conv2_bias
    # conv2_bias_1 = conv2_bn2_weight / sqrt_bn2_var * (conv2_bias - conv2_bn2_running_mean)
    # conv2_bias_upgd = conv2_bn2_bias + conv2_bias_1
    #
    # # print(conv2_bias_1[1], conv2_bias[1], conv2_bias_upgd[1])
    # # print('_____________')
    #
    # param_list = collections.OrderedDict()
    # param_list['conv1.weight'] = conv1_weight_upgd
    # param_list['conv1.bias'] = conv1_bias_upgd
    # param_list['conv2.weight'] = conv2_weight_upgd
    # param_list['conv2.bias'] = conv2_bias_upgd
    # param_list['fc3.weight'] = model_spat['decoder.fc3.weight']
    # param_list['fc3.bias'] = model_spat['decoder.fc3.bias']
    # param_list['fc4.weight'] = model_spat['decoder.fc4.weight']
    # param_list['fc4.bias'] = model_spat['decoder.fc4.bias']
    # param_list['gru.weight_ih_l0'] = model_spat['encoder.gru.weight_ih_l0']
    # param_list['gru.bias_ih_l0'] = model_spat['encoder.gru.bias_ih_l0']
    # param_list['gru.weight_hh_l0'] = model_spat['encoder.gru.weight_hh_l0']
    # param_list['gru.bias_hh_l0'] = model_spat['encoder.gru.bias_hh_l0']
    # param_list['fc.weight'] = model_spat['encoder.fc.weight']
    # param_list['fc.bias'] = model_spat['encoder.fc.bias']
    #
    # return param_list


if __name__ == "__main__":
    main()