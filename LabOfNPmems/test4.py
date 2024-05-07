# -*- coding: utf-8 -*-
"""
----------------------------------------------
Created on 2020/9/9  16:19

@author: RenXi

Email:SJTU_RX@sjtu.edu.cn
-----------------------------------------------
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃        ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset,DataLoader

x=np.linspace(1,100,100)
y=np.linspace(1,100,100)


class eval_Data_load(Dataset):
    def __init__(self,x,y):

        self.input_data = x
        self.target_data = y
        self.len = len(x)

    def __getitem__(self, item):
        return self.input_data[item], self.target_data[item]

    def __len__(self):
        return self.len



# dataset=eval_Data_load(x,y)
# dataset = DataLoader(dataset,batch_size=10,shuffle=False)
# for x,y in dataset:
#     print(x)
input_path=r"F:\0xjw_learning\home\data\sample_test\train\inputs\sample_2.pkl"



# with open(input_path, 'rb') as f:
#     input_sample = pickle.load(f)
# 
# print(input_sample.shape)


# import torch.nn as nn
# gru = nn.GRU(input_size=50, hidden_size=50, batch_first=True)
# embed = nn.Embedding(3, 50)
# 
# x = torch.LongTensor([[0, 1, 2]])
# x_embed = embed(x)
# print(embed)
# # x.size()
# torch.Size([1, 3])
# x_embed.size()
# torch.Size([1, 3, 50])
# out, hidden = gru(x_embed)
# out.size()
# torch.Size([1, 3, 50])
# hidden.size()
# torch.Size([1, 1, 50])

array=np.load(r"F:\YYC_Flow_Field_GRU\time_pred_train_target_data\inputs\input_data_0_1.npy")
print(array[0][0])