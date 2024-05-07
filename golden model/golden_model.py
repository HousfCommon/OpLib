import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from quanti import *
from gen_txt import *

class Bottleneck:
    def __init__(self, featuremap, wino_weight, w_1x1):

        #self.featuremap[32] = np.zeros((112,112))
        self.featuremap = featuremap

        #self.wino_weight = np.random.randint(-128,127,size=[33,4,4])
        self.wino_weight = wino_weight
 
        #self.wino_weight[0] = np.array([[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,2,1,2]])

        #print(self.wino_weight.shape)

        #self.w_1x1 = np.random.randint(-128,127,size=[16,33,1,1]).astype(np.float64) 
        self.w_1x1 = w_1x1
        #self.w_1x1[:,32,:,:] = np.zeros((16,1,1))

        self.result = np.empty(shape = (33,110,110))

        #self.quanitze_layer = QuantizeLayer('wino_layer', 'wino_input', 1)

    def wino_full(self):
        for i in range(self.featuremap.shape[0]):
            for j in range(int(self.featuremap.shape[1]/2)-1):
                for k in range(int(self.featuremap.shape[2]/2)-1):
                    data4x4 = self.featuremap[i,2*j:2*j+4,2*k:2*k+4]
                    weight_4x4 = self.wino_weight[i]
                    #print(i)
                    self.result[i,2*j:2*j+2,2*k:2*k+2] = self.wino_4x4(data4x4,weight_4x4)

        return self.result

    def wino_4x4(self, d_4x4, w_4x4):
        BT = np.array([[1, 0 ,-1, 0], [0, 1, 1, 0], [0, -1 , 1, 0], [0, 1, 0 ,-1]], dtype=np.int32)
        B = BT.T
        BTd = np.dot(BT,np.int32(d_4x4))
        BTdB = np.dot(BTd,B)
        #print(BTdB)
        #BTdB[BTdB > 127] = np.int32(BTdB[BTdB > 127]/4) *4
        #BTdB[BTdB < -128] = np.int32(BTdB[BTdB < -127]/4) *4
        index_max = np.argwhere(BTdB > 127)
        index_min = np.argwhere(BTdB < -128)
        for i in range(index_max.shape[0]):
            BTdB[index_max[i][0],index_max[i][1]] = np.int32(np.floor(BTdB[index_max[i][0],index_max[i][1]]/4))
        for j in range(index_min.shape[0]):
            BTdB[index_min[j][0],index_min[j][1]] = np.int32(np.floor(BTdB[index_min[j][0],index_min[j][1]]/4))
        product = BTdB*np.int32(w_4x4)

        for i in range(index_max.shape[0]):
            product[index_max[i][0],index_max[i][1]] = np.int32(product[index_max[i][0],index_max[i][1]] * 4)
        for j in range(index_min.shape[0]):
            product[index_min[j][0],index_min[j][1]] = np.int32(product[index_min[j][0],index_min[j][1]] * 4)
        AT = np.array([[1, 1, 1, 0], [0, 1 ,-1 , -1]], dtype=np.int32)
        A = AT.T
        o_2x2 = np.dot(np.dot(AT,product),A)
        return np.int32(o_2x2)
    
    def relu(self, din):
        din[din<0] = 0
        return din

    def weight_scale_gen(self, weight_blob):
        quanitze_layer = QuantizeLayer('wino_layer', 'wino_input', 1)
        quanitze_layer.quantize_weight(weight_blob, False)
        k_weight = math.log(quanitze_layer.weight_scale[0],2)
        w_scale = round(k_weight)
        return w_scale
    
    def data_scale_gen(self, blob_data):
        quanitze_layer = QuantizeLayer('wino_layer', 'wino_input', 1)
        quanitze_layer.initial_blob_max(blob_data)
        quanitze_layer.initial_blob_distubution_interval()
        quanitze_layer.initial_histograms(blob_data)
        quanitze_layer.quantize_blob()
        k_data = math.log(quanitze_layer.blob_scale, 2)
        data_scale = round(k_data)
        #print(data_scale)

        return data_scale
    
    def quanti(self,din,scale):
        quan_res = np.floor(din*(2**scale))
        quan_res = np.int32(quan_res)
        #quan_res[quan_res < 0] -= 1
        quan_res[quan_res > 127] = 127
        quan_res[quan_res < -128] = -128

        return quan_res
    
    def pw(self,din):
        din = torch.from_numpy(din).unsqueeze(0)
        w_pw = torch.from_numpy(self.w_1x1.astype(np.int32))
        pw_res = F.conv2d(din, w_pw, stride=1, padding=0)
        pw_res = pw_res.numpy().astype(np.int32) 

        return pw_res

def write_data(filename, array):
    with open(filename, 'w') as f:
        for i in range (array.shape[0]):
            f.write('[')
            f.write('\n')
            for j in range(len(array[i])):
                for k in range(len(array[i][j])):
                
                    f.write(str(array[i][j][k])+'\t')
                f.write('\n')
            f.write(']')
            f.write('\n')
            f.write('\n')


if __name__=='__main__':
    #featuremap = np.random.randint(-128,127,size=[32,112,112])
    #featuremap = np.array([[[  1,   2,   3,   4], [  5,   6,   7,   8], [  9,  10,  11,  12], [ 13,  14,  15, -127]]])
    #print(featuremap)
    #wino_weight = np.random.randint(-128,127,size=[32,4,4])
    #wino_weight = np.array([[[2, 2, 2, 2,],[1, 2, 1, 2],[1, 1, 2, 2],[0, 1, 1, 2]]])
    #print(wino_weight)
    #result = np.empty(shape = (1,2,2))

    #a = np.array([-1.2,2.1])
    #b = np.floor(a)
    #print(b)
    #print(np.int32(-2.8))


    op = Bottleneck()
    file = 'C:/CSJ/SOC/conv_ip/conv_ip/IP_of_SoC/IP_of_SoC/accv_2'
    write_data(file+'/featuremap.txt', op.featuremap)
    write_data(file+'/wino_kernel.txt', op.wino_weight)
    write_data(file+'/pw_kernel.txt', op.w_1x1)
         
    #data_scale = op.data_scale_gen(op.featuremap)
    #wino_scale = op.weight_scale_gen(op.wino_weight)
    #pw_scale = op.weight_scale_gen(op.w_1x1)
    wino_res = op.wino_full()
    write_data(file+'/wino_result.txt', wino_res.astype(np.int32))
    #print(np.max(wino_res))
    relu_res = op.relu(wino_res)
    write_data(file+'/relu_result.txt', relu_res.astype(np.int32))
    relu_scale = op.data_scale_gen(relu_res)
    quan_res_0 = op.quanti(relu_res, relu_scale)
    write_data(file+'/quan_0_result.txt', quan_res_0.astype(np.int32))
    pw_res = op.pw(quan_res_0)
    write_data(file+'/pw_result.txt', pw_res.squeeze().astype(np.int32))
    pw_scale = op.data_scale_gen(pw_res)
    output = op.quanti(pw_res, pw_scale).astype(np.int8)
    write_data(file+'/output_result.txt', output.squeeze().astype(np.int32))
    output = output[0]
    #print(np.max(output))

    txt_gen = txt_generator(op.wino_weight, op.w_1x1, op.featuremap)
    txt_gen.write(file)

    #print(output.shape)
    channel, row, col = output.shape
    wr_out = []
    for i in range(int(col/2)):
        for j in range(int(row/2)):
            wr_out += output[:,2*i:2*i+2,2*j:2*j+2].transpose(1,2,0).reshape((1,-1)).tolist()[0]
    #print(output)
    #print(wr_out)
    f=open(file+'/quan_scale.txt',"w")
    s1 = 'after relu:\t'+str(-relu_scale) + '\n'
    s2 = 'after 1x1:\t'+str(-pw_scale)
    f.write(s1+s2)
    f.close()
    

    f=open(file+'/output.txt',"w")
    hex_out = []
    bin_out = []
    for item in wr_out:
        hex_out.append('{:0>2x}'.format(item& 0xff))
        bin_out.append('{:0>8b}'.format(item& 0xff))
    for l in range(int(len(wr_out)/8)):
        line_data_h = hex_out[8*l:8*l+8]
        line_data_b = bin_out[8*l:8*l+8]
        s_h = str(line_data_h).replace('[','').replace(']','')
        #s_b = str(line_data_b).replace('[','').replace(']','')
        s_h = s_h.replace("'",'').replace(',','').replace(' ','')
        s_d = str(int(s_h,16))
        #s_b = s_b.replace("'",'').replace(',','').replace(' ','')
        #s = "{:<20}\t{:>70}".format(s_h,s_b)
        #f.write(s+'\n')
        f.write(s_h+'\n')
    f.close()







    
    '''
    result = wino_full(featuremap, wino_weight)
    print(result.shape)
    #print(result)

    result[result < 0] = 0

    k1 = 8
    quan_res = np.int32(result/2**k1)

    quan_res = torch.from_numpy(result).unsqueeze(0)
    w_1x1 = np.random.randint(-128,127,size=[16,32,1,1]).astype(np.float64) 
    print(w_1x1.shape)
    w_1x1 = torch.from_numpy(w_1x1)
    out_conv = F.conv2d(quan_res, w_1x1, stride=1, padding=0)
    out_conv = out_conv.numpy().astype(np.int32) 
    k2=8
    quan_out = np.int32(out_conv/2**k2)

    print(out_conv.shape)
    '''