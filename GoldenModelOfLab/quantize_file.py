#--just practice--
import torch
import numpy as np
import math
from nn.GoldenModelOfLab.model_test import load_spacial_weight
from nn.GoldenModelOfLab.model_test2 import load_seq2seq_weight
import nn.GoldenModelOfLab.quanti as qt


def main():
    spa_weight = load_spacial_weight()
    seq_weight = load_seq2seq_weight()
    for key in ['gru.weight_ih_l0', 'gru.bias_ih_l0', 'gru.weight_hh_l0', 'gru.bias_hh_l0']:
        # print(seq_weight[key])
        if key == 'gru.weight_ih_l0':
            print(np.max(np.abs(seq_weight[key].numpy())))
            seq_weight[key] = seq_weight[key] * 10e17 * 10e14
            print(seq_weight[key])
        else:
            seq_weight[key] = seq_weight[key] * 10e20 * 10e20
        # print(seq_weight[key])
    spa_weight = quantize_weight_file(spa_weight, 'spatial_weight', keys=['conv1.weight', 'conv1.bias',
                                                             'conv2.weight', 'conv2.bias'])
    seq_weight = quantize_weight_file(seq_weight, 'seq2seq_weight', keys=['conv1.weight', 'conv1.bias',
                                                             'conv2.weight', 'conv2.bias',
                                                             'gru.weight_ih_l0',
                                                             'gru.bias_ih_l0',
                                                             'gru.weight_hh_l0',
                                                             'gru.bias_hh_l0'])

    file_res = open('/Users/huanghuangtao/Desktop/weight_upgd.txt', 'w')
    file_res1 = open('/Users/huanghuangtao/Desktop/weight_upgd1.txt', 'w')

    print(spa_weight, file=file_res)
    print(seq_weight, file=file_res1)

    file_res.close()
    file_res1.close()

    return 0


def quantize_weight_file(weight_lib, name, keys):
    for key in keys:
        scale = qt.weight_scale_gen(weight_lib[key], key, name)
        weight_lib[key] = qt.quantize_data(weight_lib[key], scale)

    return weight_lib


if __name__ == "__main__":
    main()

