#--just practice--
import torch.nn as nn
import torch.nn.functional as F
import torch
from nn.GoldenModelOfLab.EncodeModel import EncoderModel
from nn.GoldenModelOfLab.DecoderModel import DecoderModel
import nn.GoldenModelOfLab.quanti as qt


class Seq2Seq(nn.Module):
    def __init__(self, num_features, num_hiddens, input_seq_len, pred_seq_len, out_channel,
                 batch_size, weight_lib):
        super(Seq2Seq, self).__init__()
        self.num_features = num_features
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.batch_size = batch_size
        self.encoder = EncoderModel(batch_size=batch_size, num_features=num_features, num_hiddens=num_hiddens,
                                    seq_len=input_seq_len, weight_lib=weight_lib)
        self.decoder = DecoderModel(in_channel=input_seq_len+1, out_channel=out_channel,
                                    weight_list=weight_lib,
                                    pred_seq_len=pred_seq_len, batch_size=batch_size)

    def forward(self, input_data):
        # encode process
        out, encoder_hidden = self.encoder(input_data)
        print("encoder_hidden", encoder_hidden.shape)
        input_data = input_data.view(int(input_data.shape[0] / self.input_seq_len),
                                     self.input_seq_len, -1)
        print("input_data", input_data.shape)
        encoder_input = torch.cat((input_data, encoder_hidden), dim=1)
        encoder_input_scale = qt.data_scale_gen(encoder_input, 'decoder_input', 'input_data')
        encoder_input = qt.quantize_data(encoder_input, encoder_input_scale)
        print("encoder_input", encoder_input.shape)
        # decode process
        pred_out = self.decoder(encoder_input.unsqueeze(2))
        print("pred_out", pred_out.shape)

        return pred_out


