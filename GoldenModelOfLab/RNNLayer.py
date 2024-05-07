#--just practice--
import torch
from torch import nn
import torch.nn.functional as F
import collections


class GRULayer(nn.Module):
    def __init__(self, batch_size, num_features, num_hiddens, weight_ih, bias_ih, weight_hh, bias_hh):
        super(GRULayer, self).__init__()
        self.W_ir, self.W_iz, self.W_in = torch.chunk(weight_ih, 3, dim=0)
        self.W_hr, self.W_hz, self.W_hn = torch.chunk(weight_hh, 3, dim=0)
        self.b_ir, self.b_iz, self.b_in = torch.chunk(bias_ih, 3, dim=0)
        self.b_hr, self.b_hz, self.b_hn = torch.chunk(bias_hh, 3, dim=0)
        self.batch_size = batch_size
        self.num_features = num_features
        self.num_hiddens = num_hiddens

    def forward(self, input_data, state):
        # input: [batch, len, size]
        H = state
        # print("input, Wir", input_data.dtype, self.W_ir.dtype, self.b_ir.dtype, H.dtype)
        R = torch.sigmoid(torch.matmul(input_data, self.W_ir.t()) + self.b_ir + torch.matmul(H, self.W_hr.t()) + self.b_hr)
        print("R", R.shape)
        Z = torch.sigmoid(torch.matmul(input_data, self.W_iz.t()) + self.b_iz + torch.matmul(H, self.W_hz.t()) + self.b_hz)
        N = torch.tanh(torch.matmul(input_data, self.W_in.t()) + self.b_in + R * (torch.matmul(H, self.W_hn.t()) + self.b_hn))
        H = Z * H + (1 - Z) * N
        # H_out = H.squeeze(1).unsqueeze(0)
        return H, H

    def init_gru_state(self):
        return torch.zeros(self.batch_size, 1, self.num_hiddens)
        # return torch.zeros(self.batch_size, 1, self.num_hiddens).int()


def main():
    batch_size = 2
    num_features = 10
    num_hiddens = 6
    input_data = torch.randn(2, 10)
    weight_ih = torch.randn(18, 10)
    weight_hh = torch.randn(18, 6)
    bias_ih = torch.randn(18)
    bias_hh = torch.randn(18)
    # (b1,b2,b3)= torch.chunk(weight_ih, 3, dim=0)
    # print(weight_ih)
    # print(b1)
    gru0 = GRULayer(batch_size,num_features,num_hiddens,weight_ih,bias_ih,weight_hh,bias_hh)
    H = gru0.init_gru_state()
    output, H = gru0(input_data.unsqueeze(1), H)
    print(output, output.shape)
    gru1 = nn.GRU(num_features, num_hiddens, 1, batch_first=True)
    # for name, param in gru1.named_parameters():
    #     print(name, param)

    param_list = collections.OrderedDict()
    param_list['weight_ih_l0'] = weight_ih
    param_list['weight_hh_l0'] = weight_hh
    param_list['bias_ih_l0'] = bias_ih
    param_list['bias_hh_l0'] = bias_hh

    gru1.load_state_dict(param_list)
    out, H_real = gru1(input_data.unsqueeze(1))
    print(H_real, H_real.shape)
    # exist some difference between GRULayer and nn.GRU
    return 0


if __name__ == '__main__':
    main()