import numpy as np
import matplotlib.pyplot as plt

class MacModel:
    def __init__(self, DSP_tol=1024, alg_model=None, input_size=16):
        self.DSP_tol = DSP_tol
        self.cycle_time = 0
        self.perf_eff = 0
        self.band_width = 0
        self.tol_mem = 0
        self.alg_model = alg_model
        self.input_size = input_size

        self.dsp_eff_list = []
        self.core_num_list = []
        self.core_size_list = []
        self.bandwidth_list = []
        self.bandwidth_eff_list = []
        self.tol_cycle_time_list = []
        self.mean_perf_eff_list = []
        self.tol_buf_list = []
        self.pe_buf_list = []
        self.core_buf_list = []

    def cal_mem(self, core_num, core_size):
        pe_reg = self.input_size * 4 * (core_num * core_size ** 2)
        core_buf = core_size * (core_size * 2 - 1) * (2 * self.input_size)
        self.tol_mem = (pe_reg + core_buf) / 1024 / 8
        # print("Tol Buffer size is ", self.tol_mem, "KB", "; PE Buffer size is ", pe_reg / 1024 / 8, "KB", \
        #       "; Core Buffer size is ", core_buf / 1024 / 8, "KB")
        return pe_reg / 1024 / 8, core_buf / 1024 / 8

    def cal_cycle_time_and_perf_eff(self, size=None, core_size=16, max_bandwidth=2048):
        core_num = int(self.DSP_tol / (core_size ** 2))
        # print(self.DSP_tol, core_size, core_num)
        bandwidth = core_num * core_size * 16
        if bandwidth > max_bandwidth:
            return 0, max_bandwidth+1

        Rd2 = size[0] - int(size[0] / core_size) * core_size
        Rd3 = size[2] - int(size[2] / core_size) * core_size

        if int(size[1] / core_num) < size[1] / core_num:
            b = (int(size[1] / core_num) + 1)
        else:
            b = size[1] / core_num

        cycle_time = int(size[0] / core_size) * int(size[2] / core_size) * (2 * core_size + b) * core_num \
                     + int(size[0] / core_size) * (core_size + Rd2 + b) * core_num \
                     + int(size[2] / core_size) * (core_size + Rd3 + b) * core_num \
                     + (Rd2 + Rd3 + b) * core_num
        self.cycle_time = cycle_time / core_num
        self.perf_eff = size[0] * size[1] * size[2] * 3 / (3 * (core_size ** 2) * cycle_time)

        return core_num, bandwidth

    def run(self, max_size=32):
        global bw_eff
        for i in range(max_size):
            tol_cycle_time = 0
            mean_perf_eff = 0
            bandwidth = 0
            core_num = 0
            core_size = 0
            pe_buf = 0
            core_buf = 0
            bw_eff = 0
            for size in self.alg_model:
                core_size = i + 1
                core_num, bandwidth = self.cal_cycle_time_and_perf_eff(size=size, core_size=core_size)

                if bandwidth > 2048:
                    break
                tol_cycle_time += self.cycle_time
                mean_perf_eff += self.perf_eff

                pe_buf, core_buf = self.cal_mem(core_num=core_num, core_size=core_size)

                for j in range(12):
                    bw_eff = 0
                    if 2**j < bandwidth < 2**(j + 1):
                        bw_eff = bandwidth/(2**(j+1))
                        break
                    elif bandwidth==2**j:
                        bw_eff = 1
                        break

            if bandwidth > 2048:
                continue

            # print("core num is", core_num, " size is", core_size, " bandwidth is", bandwidth, " bandwidth eff is",
            #       bw_eff, " tol_cycle_time is",
            #       tol_cycle_time, " mean_perf_eff is", mean_perf_eff / len(self.alg_model),
            #       "Tol Buffer size is ", self.tol_mem, "KB", "; PE Buffer size is ", pe_buf, "KB",
            #       "; Core Buffer size is ", core_buf, "KB")
            self.core_num_list.append(core_num)
            self.core_size_list.append(core_size)
            
            self.dsp_eff_list.append((core_size**2)*core_num/self.DSP_tol)
            self.bandwidth_list.append(bandwidth)
            self.bandwidth_eff_list.append(bw_eff)
            self.tol_cycle_time_list.append(tol_cycle_time)
            self.mean_perf_eff_list.append(mean_perf_eff)
            self.tol_buf_list.append(self.tol_mem)
            self.pe_buf_list.append(pe_buf)
            self.core_buf_list.append(core_buf)


def cal_size(seq_len,hidden_size):
    # seq_len = 100   # 100， 300， 500
    # hidden_size = 768    # 768， 3072
    
    size_0 = [seq_len, hidden_size[0], hidden_size[0]]  # a, b, c
    size_1 = [seq_len, hidden_size[0], seq_len]
    size_2 = [seq_len, seq_len, hidden_size[0]]
    size_3 = [seq_len, hidden_size[1], hidden_size[0]]
    size_4 = [seq_len, hidden_size[0], hidden_size[1]]
    return [size_0, size_0, size_0, size_1, size_2, size_3, size_4]

def main_test():
    DSP_tol = 1024
    size_tol1 = cal_size(128,[512,2048,8])
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",size_tol1)
    size_tol2 = cal_size(128,[768,3072,12])
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",size_tol2)
    size_tol3 = cal_size(256,[512,2048,8])
    size_tol4 = cal_size(256,[768,3072,12])
    size_tol5 = cal_size(512,[512,2048,8])
    size_tol6 = cal_size(512,[768,3072,12])

    mac1 = MacModel(DSP_tol=DSP_tol, alg_model=size_tol1)
    mac2 = MacModel(DSP_tol=DSP_tol, alg_model=size_tol2)
    mac3 = MacModel(DSP_tol=DSP_tol, alg_model=size_tol3)
    mac4 = MacModel(DSP_tol=DSP_tol, alg_model=size_tol4)
    mac5 = MacModel(DSP_tol=DSP_tol, alg_model=size_tol5)
    mac6 = MacModel(DSP_tol=DSP_tol, alg_model=size_tol6)
    # print(len(mac.alg_model))

    # mac2.core_buf_list
    
    mac1.run()
    # print("***********************************************************************************************\n",mac1.dsp_eff_list)
    mac2.run()
    # print("***********************************************************************************************\n",mac2.dsp_eff_list)
    mac3.run()
    mac4.run()
    mac5.run()
    mac6.run()
    
    # x = np.linspace(1,len(mac.core_size_list),len(mac.core_size_list))
# 1 
    y_01 = mac1.core_num_list
    y_11 = mac1.core_size_list #x

    y_21 = mac1.dsp_eff_list
    y_31 = mac1.bandwidth_list
    y_41 = mac1.bandwidth_eff_list
    y_51 = mac1.tol_cycle_time_list
    y_61 = mac1.mean_perf_eff_list
    y_71 = mac1.tol_buf_list
    y_81 = mac1.pe_buf_list
    y_91 = mac1.core_buf_list
#2
    y_02 = mac2.core_num_list
    y_12 = mac2.core_size_list #x

    y_22 = mac2.dsp_eff_list
    y_32 = mac2.bandwidth_list
    y_42 = mac2.bandwidth_eff_list
    y_52 = mac2.tol_cycle_time_list
    y_62 = mac2.mean_perf_eff_list
    y_72 = mac2.tol_buf_list
    y_82 = mac2.pe_buf_list
    y_92 = mac2.core_buf_list
#3
    y_03 = mac3.core_num_list
    y_13 = mac3.core_size_list #x
    
    y_23 = mac3.dsp_eff_list
    y_33 = mac3.bandwidth_list
    y_43 = mac3.bandwidth_eff_list
    y_53 = mac3.tol_cycle_time_list
    y_63 = mac3.mean_perf_eff_list
    y_73 = mac3.tol_buf_list
    y_83 = mac3.pe_buf_list
    y_93 = mac3.core_buf_list
#4
    y_04 = mac4.core_num_list
    y_14 = mac4.core_size_list #x

    y_24 = mac4.dsp_eff_list
    y_34 = mac4.bandwidth_list
    y_44 = mac4.bandwidth_eff_list
    y_54 = mac4.tol_cycle_time_list
    y_64 = mac4.mean_perf_eff_list
    y_74 = mac4.tol_buf_list
    y_84 = mac4.pe_buf_list
    y_94 = mac4.core_buf_list
#5
    y_05 = mac5.core_num_list
    y_15 = mac5.core_size_list #x
 
    y_25 = mac5.dsp_eff_list
    y_35 = mac5.bandwidth_list
    y_45 = mac5.bandwidth_eff_list
    y_55 = mac5.tol_cycle_time_list
    y_65 = mac5.mean_perf_eff_list
    y_75 = mac5.tol_buf_list
    y_85 = mac5.pe_buf_list
    y_95 = mac5.core_buf_list
#6
    y_06 = mac6.core_num_list
    y_16 = mac6.core_size_list #x

    y_26 = mac6.dsp_eff_list
    y_36 = mac6.bandwidth_list
    y_46 = mac6.bandwidth_eff_list
    y_56 = mac6.tol_cycle_time_list#
    y_66 = mac6.mean_perf_eff_list#
    y_76 = mac6.tol_buf_list
    y_86 = mac6.pe_buf_list
    y_96 = mac6.core_buf_list
    
    # plt.subplot(251)
    # print("core_size = ", y_1)
    # plt.gca().set_prop_cycle(['red',"green"])
    # print(y_95)
    # print(y_96)

    plt.figure(figsize=(24,12))
    plt.subplot(2,3,2)
    plt.plot(y_11,y_51,'r.-')
    plt.plot(y_11,y_52,'r.--')
    plt.plot(y_11,y_53,'g.-')
    plt.plot(y_11,y_54,'g.--')
    plt.plot(y_11,y_55,'b.-')
    plt.plot(y_11,y_56,'b.--')
    
    plt.legend(['[seqlen:128,hidden:512]','[seqlen:128,hidden:768]','[seqlen:256,hidden:512]',
                '[seqlen:256,hidden:768]','[seqlen:512,hidden:512]','[seqlen:512,hidden:768]'],
               loc='upper left',prop={'size':8})
    plt.xlabel("core size", fontsize=16)
    plt.ylabel("total cycle time", fontsize=16)#effeciency of digital singal processor
    x_axi = np.arange(8,32.1,2)
    plt.xticks(x_axi, fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(2, 3, 1)
    plt.plot(y_11, y_21, 'r.-')
    plt.plot(y_11, y_22, 'r.--')
    plt.plot(y_11, y_23, 'g.-')
    plt.plot(y_11, y_24, 'g.--')
    plt.plot(y_11, y_25, 'b.-')
    plt.plot(y_11, y_26, 'b.--')

    plt.legend(['[seqlen:128,hidden:512]', '[seqlen:128,hidden:768]', '[seqlen:256,hidden:512]',
                '[seqlen:256,hidden:768]', '[seqlen:512,hidden:512]', '[seqlen:512,hidden:768]'],
               loc='upper right', prop={'size': 8})
    plt.xlabel("core size", fontsize=16)
    plt.ylabel("efficiency of DSP", fontsize=16)  # effeciency of digital singal processor
    x_axi = np.arange(8, 32.1, 2)
    plt.xticks(x_axi, fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(2, 3, 3)
    plt.plot(y_11, y_61, 'r.-')
    plt.plot(y_11, y_62, 'r.--')
    plt.plot(y_11, y_63, 'g.-')
    plt.plot(y_11, y_64, 'g.--')
    plt.plot(y_11, y_65, 'b.-')
    plt.plot(y_11, y_66, 'b.--')

    plt.legend(['[seqlen:128,hidden:512]', '[seqlen:128,hidden:768]', '[seqlen:256,hidden:512]',
                '[seqlen:256,hidden:768]', '[seqlen:512,hidden:512]', '[seqlen:512,hidden:768]'],
               loc='upper left', prop={'size': 8})
    plt.xlabel("core size", fontsize=16)
    plt.ylabel("mean performance efficiency", fontsize=16)  # effeciency of digital singal processor
    x_axi = np.arange(8, 32.1, 2)
    plt.xticks(x_axi, fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(2, 3, 4)
    plt.plot(y_11, y_31, 'r.-')
    plt.plot(y_11, y_32, 'r.--')
    plt.plot(y_11, y_33, 'g.-')
    plt.plot(y_11, y_34, 'g.--')
    plt.plot(y_11, y_35, 'b.-')
    plt.plot(y_11, y_36, 'b.--')

    # plt.legend(['[seqlen:128,hidden:512]', '[seqlen:128,hidden:768]', '[seqlen:256,hidden:512]',
    #             '[seqlen:256,hidden:768]', '[seqlen:512,hidden:512]', '[seqlen:512,hidden:768]'],
    #            loc='upper left', prop={'size': 10})
    plt.xlabel("core size", fontsize=16)
    plt.ylabel("bandwidth", fontsize=16)  # effeciency of digital singal processor
    x_axi = np.arange(8, 32.1, 2)
    plt.xticks(x_axi, fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(2, 3, 5)
    plt.plot(y_11, y_41, 'r.-')
    plt.plot(y_11, y_42, 'r.--')
    plt.plot(y_11, y_43, 'g.-')
    plt.plot(y_11, y_44, 'g.--')
    plt.plot(y_11, y_45, 'b.-')
    plt.plot(y_11, y_46, 'b.--')

    # plt.legend(['[seqlen:128,hidden:512]', '[seqlen:128,hidden:768]', '[seqlen:256,hidden:512]',
    #             '[seqlen:256,hidden:768]', '[seqlen:512,hidden:512]', '[seqlen:512,hidden:768]'],
    #            loc='upper left', prop={'size': 10})
    plt.xlabel("core size", fontsize=16)
    plt.ylabel("usage of bandwidth", fontsize=16)  # effeciency of digital singal processor
    x_axi = np.arange(8, 32.1, 2)
    plt.xticks(x_axi, fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(2, 3, 6)
    plt.plot(y_11, y_71, 'r.-')
    plt.plot(y_11, y_72, 'r.--')
    plt.plot(y_11, y_73, 'g.-')
    plt.plot(y_11, y_74, 'g.--')
    plt.plot(y_11, y_75, 'b.-')
    plt.plot(y_11, y_76, 'b.--')

    # plt.legend(['[seqlen:128,hidden:512]', '[seqlen:128,hidden:768]', '[seqlen:256,hidden:512]',
    #             '[seqlen:256,hidden:768]', '[seqlen:512,hidden:512]', '[seqlen:512,hidden:768]'],
    #            loc='upper left', prop={'size': 10})
    plt.xlabel("core size", fontsize=16)
    plt.ylabel("total buffer", fontsize=16)  # effeciency of digital singal processor
    x_axi = np.arange(8, 32.1, 2)
    plt.xticks(x_axi, fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()
    
# def main():
#     # size:
#     # [n, 768, 64] * 3
#     # [n, 64, n] * 1
#     # [n, n, 64] * 1
#     n = 200
#     size_0 = [n, 768, 64]  # a, b, c
#     size_1 = [n, 64, n]
#     size_2 = [n, n, 64]
#     size_3 = [n, 3096, 768]
#     size_4 = [n, 768, 3096]
#     DSP_tol = 1024

#     tol_mem = (768 * 64 * 4 + n * 64 * 3 + n * n + n * 64 + n * 768 + 6 * 768) * 16 / 8 / 1024
#     print("SRAM size is ", tol_mem, "KB")

#     for i in range(32):
#         tol_cycle_time = 0
#         mean_perf_eff = 0
#         for size in [size_0, size_0, size_0, size_1, size_2]:
#             L = i + 1
#             N = int(DSP_tol / (L ** 2))

#             Rd2 = size[0] - int(size[0] / L) * L
#             Rd3 = size[2] - int(size[2] / L) * L

#             bandwidth = N * L * 16
#             if bandwidth > 2048:
#                 continue

#             if int(size[1] / N) < size[1] / N:
#                 b = (int(size[1] / N) + 1)
#             else:
#                 b = size[1] / N

#             cycle_time = int(size[0] / L) * int(size[2] / L) * (2 * L + b) * N \
#                          + int(size[0] / L) * (L + Rd2 + b) * N \
#                          + int(size[2] / L) * (L + Rd3 + b) * N \
#                          + (Rd2 + Rd3 + b) * N

#             cycle_time_min = int(size[0] / L) * int(size[2] / L) * (2 * L + b) * N \
#                              + int(size[0] / L) * (b) * N \
#                              + int(size[2] / L) * (b) * N \
#                              + (b) * N

#             cycle_time_mean = cycle_time / N
#             tol_cycle_time += cycle_time_mean

#             perf_eff = size[0] * size[1] * size[2] * 3 / (3 * (L ** 2) * cycle_time)
#             mean_perf_eff += perf_eff

#             # print("core num is", N, " size is", L, "bandwidth is", bandwidth, " cycletime is", cycle_time_mean,
#             # " performance eff is", perf_eff)

#         if bandwidth > 2048:
#             continue

#         memory_size = 32 * (L ** 2) * N + (L ** 2) * 16 * 2

#         print("core num is", N, " size is", L, " bandwidth is", bandwidth, " tol_cycle_time is",
#               tol_cycle_time, " mean_perf_eff is", mean_perf_eff / 5, " memory size is", memory_size)


if __name__ == "__main__":
    main_test()

