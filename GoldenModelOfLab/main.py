#--just practice--
import numpy as np
import torch
from nn.GoldenModelOfLab.SpacialModel import GoldenSpatialModel
from nn.GoldenModelOfLab.Seq2Seq import Seq2Seq
from nn.GoldenModelOfLab.model_test import load_spacial_weight
from nn.GoldenModelOfLab.model_test2 import load_seq2seq_weight
from nn.LabOfNPmems.seq2seq_new import Seq2seq_new
from nn.LabOfNPmems.spatialmodel import SpatialModel
import nn.GoldenModelOfLab.quanti as qt
from nn.GoldenModelOfLab.quantize_file import quantize_weight_file
from nn.GoldenModelOfLab.file_gen import file_generator

def main():
    gen_flag = 1
    path = "/Users/huanghuangtao/Desktop/memory"
    gen = file_generator('txt_gen')
    param_list_spa = load_spacial_weight()
    param_list_seq = load_seq2seq_weight()

    # for key in ['gru.weight_ih_l0', 'gru.bias_ih_l0', 'gru.weight_hh_l0', 'gru.bias_hh_l0']:
    #     # print(seq_weight[key])
    #     if key == 'gru.weight_ih_l0':
    #         param_list_seq[key] = param_list_seq[key] * 10e17 * 10e14
    #     else:
    #         param_list_seq[key] = param_list_seq[key] * 10e20 * 10e20
    #     # print(seq_weight[key])
    param_list_spa = quantize_weight_file(param_list_spa,
                                          'spatial_weight', keys=['conv1.weight', 'conv1.bias',
                                          'conv2.weight', 'conv2.bias'])
    param_list_seq = quantize_weight_file(param_list_seq,
                                          'seq2seq_weight', keys=['conv1.weight', 'conv1.bias',
                                          'conv2.weight', 'conv2.bias'
                                          # ,'gru.weight_ih_l0',
                                          # 'gru.bias_ih_l0',
                                          # 'gru.weight_hh_l0',
                                          # 'gru.bias_hh_l0'
                                          ])

    if gen_flag:
        for key in ['conv1.weight',  'conv2.weight']:
            gen.write(path, 'spa_'+key, gen.weight_sram_gen(param_list_spa[key]))
            gen.write(path, 'seq_'+key, gen.weight_sram_gen(param_list_seq[key]))
        for key in ['conv1.bias', 'conv2.bias']:
            gen.write(path, 'spa_' + key, gen.bias_sram_gen(param_list_spa[key]))
            gen.write(path, 'seq_' + key, gen.bias_sram_gen(param_list_seq[key]))

    checkpoint = torch.load('/Users/huanghuangtao/Desktop/check_point_50', map_location='cpu')
    input_seq_len = 10
    pred_seq_len = 25
    batch_size = 1

    with torch.no_grad():
        model_spatial = GoldenSpatialModel(in_channels=5, out_num=1053, weight_lib=param_list_spa, batch_size=batch_size*input_seq_len)
        model_time = Seq2Seq(num_features=1053, num_hiddens=512, input_seq_len=input_seq_len,
                             pred_seq_len=pred_seq_len, out_channel=1053,
                             batch_size=batch_size, weight_lib=param_list_seq)

        ref_spatial = SpatialModel(num_input_channels=5, out_num=1053,
                                   dropout_p=0.1)
        ref_spatial.load_state_dict(checkpoint['model_spatial'])
        ref_spatial.eval()
        ref_time = Seq2seq_new(num_features=1053, hidden_size=512, input_seq_len=input_seq_len,
                               pred_seq_len=pred_seq_len, batch_size=batch_size)
        ref_time.load_state_dict(checkpoint['model_time'])
        ref_time.eval()

        input_data = torch.randn(10, 5, 17, 31)

        ret1 = ref_spatial(input_data)
        ret = ref_time(ret1, device='cpu')

        input_scale = qt.data_scale_gen(input_data, 'spatial_conv1', 'input_data')
        input_data = qt.quantize_data(input_data, input_scale)

        if gen_flag:
            gen.write(path, 'input_data', gen.data_sram_gen(input_data))

        res1 = model_spatial(input_data)
        print("spatial_data", res1.shape)

        res1_scale = qt.data_scale_gen(res1, 'time_gru', 'input_data')
        res1_q = qt.quantize_data(res1, res1_scale)
        print("spatial_result", res1_q)
        res = model_time(res1_q.float())

        # model testing case
        # print(res.shape, ret.shape)
        file_res = open('/Users/huanghuangtao/Desktop/weight_upgd.txt', 'w')
        file_res1 = open('/Users/huanghuangtao/Desktop/weight_upgd1.txt', 'w')
        print(res.numpy(), file=file_res)
        print(ret.numpy(), file=file_res1)
        # p = (ret - res) / ret
        # p = p.numpy()
        # p = np.sum(np.abs(p)) / (1053*pred_seq_len*batch_size)
        # print("result:", p)

        file_res.close()
        file_res1.close()


if __name__ == "__main__":
    main()