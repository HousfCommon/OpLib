#--just practice--
import numpy as np
import torch
import collections
import nn.LabOfNPmems.seq2seq_new as seq
import nn.GoldenModelOfLab.Seq2Seq as seq2
from nn.GoldenModelOfLab.model_test2 import load_seq2seq_weight


def main():
    param_list = load_seq2seq_weight()
    checkpoint = torch.load('/Users/huanghuangtao/Desktop/check_point_50', map_location='cpu')
    input_seq_len = 10
    pred_seq_len = 25
    with torch.no_grad():
        model = seq.Seq2seq_new(num_features=1053, hidden_size=512,
                                input_seq_len=input_seq_len, pred_seq_len=pred_seq_len,
                                batch_size=16)
        model.load_state_dict(checkpoint['model_time'])
        model.eval()

        model2 = seq2.Seq2Seq(num_features=1053, num_hiddens=512, input_seq_len=input_seq_len,
                              pred_seq_len=pred_seq_len, out_channel=1053,
                              batch_size=16, weight_lib=param_list)

        input_data = torch.randn(16*10,1053)
        res = model2(input_data)
        ret = model(input_data, device='cpu')
        print(ret.shape, res.shape)

        file_res = open('/Users/huanghuangtao/Desktop/weight_upgd.txt', 'w')
        file_res1 = open('/Users/huanghuangtao/Desktop/weight_upgd1.txt', 'w')
        print(res.numpy(), file=file_res)
        print(ret.numpy(), file=file_res1)
        p = (ret - res) / ret
        p = p.numpy()
        p = np.sum(np.abs(p)) / 1053
        print("result:", p)

        file_res.close()
        file_res1.close()


if __name__ == "__main__":
    main()