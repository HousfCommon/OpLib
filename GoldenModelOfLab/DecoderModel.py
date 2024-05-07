#--just practice--
import torch.nn as nn
import torch.nn.functional as F
import torch
import nn.GoldenModelOfLab.quanti as qt
import nn.GoldenModelOfLab.Layer as L


class DecoderModel(nn.Module):
    def __init__(self, in_channel, out_channel, weight_list, pred_seq_len, batch_size):
        super(DecoderModel, self).__init__()
        self.pred_seq_len = pred_seq_len
        self.batch_size = batch_size
        self.out_num = out_channel
        self.conv1 = L.Conv2DLayer(in_channels=in_channel, out_channels=64, kernel_size=5, padding=2,
                                   weight=weight_list['conv1.weight'] , bias=weight_list['conv1.bias'])
        self.conv2 = L.Conv2DLayer(in_channels=64, out_channels=64, kernel_size=5, padding=2,
                                   weight=weight_list['conv2.weight'] , bias=weight_list['conv2.bias'] )
        self.fc3 = L.LinearLayer(in_features=64*65, out_features=5120,
                                 weight=weight_list['fc3.weight'] , bias=weight_list['fc3.bias'])
        self.fc4 = L.LinearLayer(in_features=5120, out_features=out_channel*self.pred_seq_len,
                                 weight=weight_list['fc4.weight'], bias=weight_list['fc4.bias'])

    def forward(self, input_data):
        y = F.relu(self.conv1(input_data))
        print("y", y.shape)
        y = L.max_pool2d(y, kernel_size=(4,1), stride=(4,1), padding=0)
        y_scale = qt.data_scale_gen(y, 'decoder_conv2', 'input_data')
        y = qt.quantize_data(y, y_scale)
        y = F.relu(self.conv2(y))
        y = L.max_pool2d(y, kernel_size=(4,1), stride=(4,1), padding=0)

        y = y.reshape(self.batch_size, -1)
        y = self.fc3(y)
        y_pred = self.fc4(y)

        return y_pred.reshape(self.batch_size, self.pred_seq_len, self.out_num)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():

    return 0

if __name__ == "__main__":
    main()