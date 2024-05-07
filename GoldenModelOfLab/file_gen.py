#--just practice--
import numpy as np
import torch


class file_generator:
    def __init__(self, name):
        self.name = name

    def weight_sram_gen(self, weight):
        if torch.is_tensor(weight):
            weight = weight.numpy()
        N, C, H, W = weight.shape
        weight = weight.reshape((1, -1), order='C')
        # print(wino_reshape)

        weight_list = weight.tolist()[0]
        # print(len(wino_list))
        # weight_txt = [[] for _ in range(4)]
        weight_txt = []
        zero_list = [0 for _ in range(7)]
        # print(wino_txt)

        for i in range(int(len(weight_list) / (H*W))):
            weight_txt.extend(weight_list[(H*W) * i:(H*W) * i + 8])
            weight_txt.extend(weight_list[(H*W) * i + 8:(H*W) * i + 16])
            weight_txt.extend(weight_list[(H*W) * i + 16:(H*W) * i + 24])
            weight_txt.append(weight_list[(H*W) * i + 24])
            weight_txt.extend(zero_list)
        # print(wino_txt)
        return weight_txt

    def bias_sram_gen(self, bias):
        if torch.is_tensor(bias):
            bias = bias.numpy()
        # N, C, H, W = bias.shape
        bias = bias.reshape((1, -1), order='C')
        bias_list = bias.tolist()[0]
        # bias_txt = []

        # for i in range(int(len(bias_list) / (H*W))):
        #     bias_txt.extend()

        return bias_list

    def data_sram_gen(self, data):
        if torch.is_tensor(data):
            data = data.numpy()
        H, W = data.shape[-2], data.shape[-1]
        data = data.reshape((1, -1), order='C')
        data_list = data.tolist()[0]
        err = 8 - (H*W) % 8
        zero_list = [0 for _ in range(err)]

        data_txt = []

        for i in range(int(len(data_list) / (H*W))):
            data_txt.extend(data_list[(H*W)*i:(H*W)*(i+1)])
            data_txt.extend(zero_list)

        return data_txt

    def wr_txt(self, sram_txt, filename):
        if(len(sram_txt)%8 != 0):
            print('error!!!!')

        f=open(filename,"w")
        for l in range(int(len(sram_txt)/8)):
            line_data = sram_txt[8*l:8*l+8]
            s = str(line_data).replace('[','').replace(']','')
            s = s.replace("'",'').replace(',','').replace(' ','') + '\n'
            f.write(s)
        f.close()

    def write(self, path, name, data_txt):
        # weight_txt = self.weight_sram_gen()
        # bias_txt = self.bias_sram_gen()
        # data_txt = self.data_sram_gen()
        #
        # sram_txt = [[] for _ in range(3)]
        # sram_txt[0] = weight_txt
        # sram_txt[1] = bias_txt
        # sram_txt[2] = data_txt
        # for i in range(3):
        #     hex_sram = []
        #     for item in sram_txt[i]:
        #         hex_sram.append('{:02x}'.format(int(item)))
        #     #print(sram_txt[i])
        #     #print(hex_sram)
        #     self.wr_txt(hex_sram, path+'/memory_'+str(i)+'.txt')
        hex_sram = []
        for item in data_txt:
            hex_sram.append('{:0>2x}'.format(int(item)&0xff))
        # print(sram_txt[i])
        # print(hex_sram)
        self.wr_txt(hex_sram, path + '/memory_' + name + '.txt')


def main():
    path = "/Users/huanghuangtao/Desktop"
    weight = torch.ones(64,5,5,5).int()
    bias = torch.ones(64).int()
    data = torch.ones(31,17,5).int()
    gen = file_generator(name='test')
    # weight_txt = gen.weight_sram_gen(weight=weight)
    # gen.write(path,'test',weight_txt)
    bias_txt = gen.bias_sram_gen(bias)
    gen.write(path, 'bias',bias_txt)
    # data_txt = gen.data_sram_gen(data)
    # gen.write(path,'data',data_txt)
    # print(data.shape[-1])
    return 0


if __name__ == "__main__":
    main()