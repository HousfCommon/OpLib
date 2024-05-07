import numpy as np


class txt_generator:

    def __init__(self, wino_weight, w_1x1, featuremap):
        self.wino_weight = wino_weight
        self.w_1x1 = w_1x1
        self.featuremap = featuremap
        
    def wino_sram_gen(self):
        wino_reshape = self.wino_weight.reshape((1,-1), order='C')
        #print(wino_reshape)

        wino_list = wino_reshape.tolist()[0]
        #print(len(wino_list))
        wino_txt = [[] for _ in range(6)]
        #print(wino_txt)

        for i in range(int(len(wino_list)/24)):
            wino_txt[0].extend(wino_list[24*i:24*i+4])
            wino_txt[1].extend(wino_list[24*i+4:24*i+8])
            wino_txt[2].extend(wino_list[24*i+8:24*i+12])
            wino_txt[3].extend(wino_list[24*i+12:24*i+16])
            wino_txt[4].extend(wino_list[24*i+16:24*i+20])
            wino_txt[5].extend(wino_list[24*i+20:24*i+24])
        #print(wino_txt)

        return wino_txt

    def pw_sram_gen(self):
        w_pw = self.w_1x1.astype(np.int32)
        pw_reshape = w_pw.transpose(1,0,2,3).reshape((1,-1), order='C')
        #print(pw_reshape)

        pw_list = pw_reshape.tolist()[0]
        #print(len(pw_list))
        pw_txt = [[] for _ in range(6)]
        #print(pw_txt)

        for i in range(int(len(pw_list)/24)):
            pw_txt[0].extend(pw_list[24*i:24*i+4])
            pw_txt[1].extend(pw_list[24*i+4:24*i+8])
            pw_txt[2].extend(pw_list[24*i+8:24*i+12])
            pw_txt[3].extend(pw_list[24*i+12:24*i+16])
            pw_txt[4].extend(pw_list[24*i+16:24*i+20])
            pw_txt[5].extend(pw_list[24*i+20:24*i+24])
        #print(pw_txt)

        return pw_txt

    def restoredata(self):
        channel = self.featuremap.shape[0]
        row = self.featuremap.shape[1]
        col = self.featuremap.shape[2]
        featuremap_t = self.featuremap.transpose(1,2,0)
        # print(featuremap[0:1,0,0].shape)
        # print(featuremap.shape)

        sram = np.zeros((6,int(col/4),int(row/2),int(channel*4/3)))
        for i in range(int(col/4)):
            for j in range(int(row/2)):
                for k in range(int(channel/3)):
                    sram[0,i,j,k*4:k*4+4] = featuremap_t[4*i:4*i+2,2*j:2*j+2,3*k].reshape((4,))
                    sram[1,i,j,k*4:k*4+4] = featuremap_t[4*i+2:4*i+4,2*j:2*j+2,3*k].reshape((4,))
                    sram[2,i,j,k*4:k*4+4] = featuremap_t[4*i:4*i+2,2*j:2*j+2,3*k+1].reshape((4,))
                    sram[3,i,j,k*4:k*4+4] = featuremap_t[4*i+2:4*i+4,2*j:2*j+2,3*k+1].reshape((4,))
                    sram[4,i,j,k*4:k*4+4] = featuremap_t[4*i:4*i+2,2*j:2*j+2,3*k+2].reshape((4,))
                    sram[5,i,j,k*4:k*4+4] = featuremap_t[4*i+2:4*i+4,2*j:2*j+2,3*k+2].reshape((4,))

        sram = sram.reshape(6,-1)
        return sram.astype(np.int32)


    def wr_txt(self, sram_txt, filename):
        if(len(sram_txt)%4 != 0):
            print('error!!!!')

        f=open(filename,"w")
        for l in range(int(len(sram_txt)/4)):
            line_data = sram_txt[4*l:4*l+4]
            s = str(line_data).replace('[','').replace(']','')
            s = s.replace("'",'').replace(',','').replace(' ','') + '\n'
            f.write(s)
        f.close()
    
    def write(self, path):
        wino_txt = self.wino_sram_gen()
        pw_txt = self.pw_sram_gen()
        data_txt = self.restoredata().tolist()

        sram_txt = [[] for _ in range(6)]

        for i in range(6):
            sram_txt[i] = wino_txt[i] + pw_txt[i] + data_txt[i]
            hex_sram = []
            for item in sram_txt[i]:
                #print(item)
                #hex_sram.append('%02x'%(item & 0xff))
                hex_sram.append('{:0>2x}'.format(item& 0xff))
                #hex_sram.append('{:0>8b}'.format(item& 0xff))
            #print(sram_txt[i])
            #print(hex_sram)
            self.wr_txt(hex_sram, path+'\memory_'+str(i)+'.txt')


if __name__=='__main__':
    #wino_weight = np.random.randint(-128,127,size=[32,4,4])
    featuremap = np.random.randint(-128,127,size=[33,112,112])
    featuremap[32] = np.zeros((112,112))
    wino_weight = np.random.randint(-128,127,size=[33,4,4])
    wino_weight[32] = np.zeros((4,4))
    w_1x1 = np.random.randint(-128,127,size=[16,33,1,1]).astype(np.float64) 
    w_1x1[:,32,:,:] = np.zeros((16,1,1))
    
    txt_gen = txt_generator(wino_weight, w_1x1, featuremap)
    txt_gen.write('C:\CSJ\SOC\conv_ip\conv_ip\IP_of_SoC\IP_of_SoC\Golden_Model')

'''
    #print(wino_weight)
    wino_txt = wino_sram_gen(wino_weight)
    pw_txt = pw_sram_gen(w_1x1)
    data_txt = restoredata(featuremap).tolist()

    sram_txt = [[] for _ in range(6)]
    for i in range(6):
        sram_txt[i] = wino_txt[i] + pw_txt[i] + data_txt[i]
        hex_sram = []
        for item in sram_txt[i]:
            #print(item)
            #hex_sram.append('%02x'%(item & 0xff))
            hex_sram.append('{:0>2x}'.format(item& 0xff))
            #hex_sram.append('{:0>8b}'.format(item& 0xff))
        #print(sram_txt[i])
        #print(hex_sram)
        wr_txt(hex_sram, 'C:\CSJ\SOC\conv_ip\conv_ip\IP_of_SoC\IP_of_SoC\Golden_Model\memory_'+str(i)+'.txt')
'''