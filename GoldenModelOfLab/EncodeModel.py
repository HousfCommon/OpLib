#--just practice--
import torch.nn as nn
import torch.nn.functional as F
import torch
from nn.GoldenModelOfLab.RNNLayer import GRULayer
from nn.GoldenModelOfLab.Layer import LinearLayer
import collections
import nn.LabOfNPmems.seq2seq_new as seq


class EncoderModel(nn.Module):
    def __init__(self, batch_size, num_features, num_hiddens, seq_len, weight_lib):
        super(EncoderModel, self).__init__()
        self.gru = GRULayer(batch_size, num_features, num_hiddens,
                            weight_lib['gru.weight_ih_l0'],
                            weight_lib['gru.bias_ih_l0'],
                            weight_lib['gru.weight_hh_l0'],
                            weight_lib['gru.bias_hh_l0'])
        self.fc = LinearLayer(num_hiddens, num_features, weight_lib['fc.weight'],
                              weight_lib['fc.bias'])
        self.seq_len = seq_len
        self.batch_size = batch_size

    def forward(self, input_data):
        input_data = input_data.view(self.batch_size,
                                     self.seq_len, -1)
        # print(input_data[:, 1, :])
        print("input_data", input_data.shape)
        Ht = self.gru.init_gru_state()

        for i in range(self.seq_len):
            oh, Ht = self.gru(input_data[:, i, :].unsqueeze(1), self.gru.init_gru_state())
            Ht_trans = F.relu(oh)
            # print("i_model",i ,Ht_trans)

        print("Ht", Ht.shape)
        # Ht_trans = F.relu(oh)
        encoder_hidden = F.relu(self.fc(Ht))
        # res = encoder_hidden.squeeze(1).unsqueeze(0)
        return Ht_trans, encoder_hidden


def main():
    batch_size = 4
    num_features = 5
    num_hiddens = 3
    seq_len = 2
    input_data = torch.randn(4*seq_len, 5)
    weight_ih = torch.randn(9, 5)
    weight_hh = torch.randn(9, 3)
    bias_ih = torch.randn(9)
    bias_hh = torch.randn(9)
    fc_weight = torch.randn(5, 3)
    fc_bias = torch.randn(5)

    param_list = collections.OrderedDict()
    param_list['gru.weight_ih_l0'] = weight_ih
    param_list['gru.weight_hh_l0'] = weight_hh
    param_list['gru.bias_ih_l0'] = bias_ih
    param_list['gru.bias_hh_l0'] = bias_hh
    param_list['fc.weight'] = fc_weight
    param_list['fc.bias'] = fc_bias

    with torch.no_grad():
        encoder = EncoderModel(batch_size=batch_size, num_features=num_features, num_hiddens=num_hiddens, seq_len=seq_len, weight_lib=param_list)
        H1, H = encoder(input_data)
        # input_data1 = input_data.view(3,
        #                               int(input_data.shape[0] / 3), -1)
        input_data_2 = input_data.view(int(input_data.shape[0] / seq_len), seq_len, -1)
        # encoder2 = seq.EncoderModel(num_features=num_features, hidden_size=num_hiddens, layer=1, seq_len=seq_len)
        # encoder2.eval()
        # encoder2.load_state_dict(param_list)
        # for i in range(seq_len):
        #     z, H_real = encoder2(input_data_2[:, i, :].unsqueeze(1), 'cpu')
        #     print("i", i, z)
        # # print(input_data_2[:,1,:])

        # print("H_real", H_real.shape)
        # print(H_real)

        print("H", H.shape)
        print(H)
    return 0


if __name__ == "__main__":
    main()