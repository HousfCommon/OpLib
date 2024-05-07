# --just practice--
import numpy as np
import matplotlib.pyplot as plt


class EncoderModel:
    def __init__(self, N_in, Input, Hidden, Output=None):
        self.N_in = N_in
        self.Input = Input
        self.Output = Output
        self.Hidden = Hidden

    def cal_mac(self):
        Num_k = self.N_in * self.Input[1] * self.Input[0] * self.Hidden[0] * 3
        Num_s = self.N_in * (self.Input[1] ** 2) * self.Hidden[0] * 2
        Num_c = self.N_in * self.Input[1] * self.Hidden[0] * self.Hidden[1]
        Num_f = self.Input[1] * self.Hidden[1] * self.Hidden[2] + self.Input[1] * self.Hidden[3] * self.Hidden[2]
        Num = [Num_k, Num_s, Num_c, Num_f, Num_k + Num_s + Num_c + Num_f]
        return Num

    def cal_nvo(self):
        Num_sfm = self.N_in * (self.Input[1] ** 2)
        Num_ln = self.N_in * self.Input[1] * self.Hidden[0] + self.Input[1] * self.Hidden[3]
        Num_gl = self.Input[1] * self.Hidden[2]
        Num = [Num_sfm, Num_ln, Num_gl]
        return Num

    def cal_weight(self):
        Num_tol = self.N_in * self.Input[0] * self.Hidden[0] * 3 + \
                  self.N_in * self.Hidden[0] * self.Hidden[1] + \
                  self.Hidden[1] * self.Hidden[2] + \
                  self.Hidden[2] * self.Hidden[3]
        return [self.N_in * self.Input[0] * self.Hidden[0] * 3, self.N_in * self.Hidden[0] * self.Hidden[1],
                self.Hidden[1] * self.Hidden[2], self.Hidden[2] * self.Hidden[3]]


def main():
    # configure setting
    seq_len = [128, 256, 512]
    hidden_size = [[512, 2048], [768, 3072]]

    # model configure
    encoder = []

    for i in range(len(seq_len)):
        for j in range(len(hidden_size)):
            N_in = hidden_size[j][0] / 64
            Input = [hidden_size[j][0], seq_len[i], 1]
            Hidden = [hidden_size[j][0] / N_in, hidden_size[j][0], hidden_size[j][1], hidden_size[j][0]]
            print(Input, Hidden)

            encoder.append(EncoderModel(N_in=N_in, Input=Input, Hidden=Hidden))

    plt.subplot(1, 3, 1)
    x = np.array([0, 1, 2, 3, 4, 5])
    y_mac = np.zeros((4, 6))
    for i in x:
        y_mac[0][i] = encoder[i].cal_mac()[0]
        y_mac[1][i] = encoder[i].cal_mac()[1]
        y_mac[2][i] = encoder[i].cal_mac()[2]
        y_mac[3][i] = encoder[i].cal_mac()[3]

    plt.bar(x, y_mac[0], width=0.18, color='darkseagreen', label='key,query,value')
    x = x + 0.2
    plt.bar(x, y_mac[2], width=0.18, color='red', label='concat')
    x = x + 0.2
    plt.bar(x, y_mac[1], width=0.18, color='salmon', label='matmul,score')
    x = x + 0.2
    plt.bar(x, y_mac[3], width=0.18, color='navy', label='feed forward')

    plt.xlabel("Configuration:[Sequence Length, Hidden Size]")
    plt.ylabel("Operation Num")

    labels = ["[128,512]", "[128,768]",
              "[256,512]", "[256,768]",
              "[512,512]", "[512,768]"]

    plt.xticks(x-0.3, labels)
    plt.legend()
    plt.title("MAC Operation")

    plt.subplot(1, 3, 2)
    x = np.array([0, 1, 2, 3, 4, 5])
    y_nvo = np.zeros((3, 6))
    for i in x:
        y_nvo[0][i] = encoder[i].cal_nvo()[0]
        y_nvo[1][i] = encoder[i].cal_nvo()[1]
        y_nvo[2][i] = encoder[i].cal_nvo()[2]

    plt.bar(x, y_nvo[0], width=0.28, color='darkseagreen', label='softmax')
    x = x + 0.3
    plt.bar(x, y_nvo[1], width=0.28, color='red', label='layernorm')
    x = x + 0.3
    plt.bar(x, y_nvo[2], width=0.28, color='navy', label='gelu')

    plt.xlabel("Configuration:[Sequence Length, Hidden Size]")
    plt.ylabel("Operation Num")

    labels = ["[128,512]", "[128,768]",
              "[256,512]", "[256,768]",
              "[512,512]", "[512,768]"]

    plt.xticks(x - 0.3, labels)
    plt.legend()
    plt.title("Nonlinear Vector Operation")
    plt.subplot(1, 3, 3)

    x = np.array([0, 1, 2, 3, 4, 5])
    y_mac = np.zeros((4, 6))
    for i in x:
        y_mac[0][i] = encoder[i].cal_weight()[0]
        y_mac[1][i] = encoder[i].cal_weight()[1]
        y_mac[2][i] = encoder[i].cal_weight()[2]
        y_mac[3][i] = encoder[i].cal_weight()[3]

    plt.bar(x[0:2], y_mac[0][0:2], width=0.15, color='darkseagreen', label='key,query,value')
    x = x + 0.2
    plt.bar(x[0:2], y_mac[1][0:2], width=0.15, color='red', label='concat')
    x = x + 0.2
    plt.bar(x[0:2], y_mac[2][0:2], width=0.15, color='salmon', label='feed forward Layer1')
    x = x + 0.2
    plt.bar(x[0:2], y_mac[3][0:2], width=0.15, color='navy', label='feed forward Layer2')

    plt.xlabel("Configuration:[Hidden Size]")
    plt.ylabel("Weight Num")

    labels = ["512", "768"]

    plt.xticks(x[0:2] - 0.3, labels)
    plt.legend()
    plt.title("Number of Weight")


    plt.show()



    # fig = plt.figure()
    # plt1 = fig.add_subplot(1, 2, 1)
    #
    # plt1.bar(x, y_mac[0], width=0.5, color='blue', label='key,query,value')
    # plt1.bar(x, y_mac[2], width=0.5, color='red', label='concat')
    # plt1.bar(x, y_mac[1], width=0.5, color='black', label='matmul,score')
    #
    # plt1.set_xlabel("Configuration:[Sequence Length, Hidden Size]")
    # plt1.set_ylabel("Operation Num")
    # labels = ["[128,512]", "[128,768]",
    #           "[256,512]", "[256,768]",
    #           "[512,512]", "[512,768]"]
    # # plt1.set_xticks(labels)
    # plt1.legend()
    #
    # plt.xticks(x, labels)
    #
    # plt2 = fig.add_subplot(1, 2, 2)
    # plt2.bar(x, y_mac[3], width=0.1, color='green', label='feed forward')
    # plt2.set_xlabel("Configuration:[Sequence Length, Hidden Size]")
    # plt2.set_ylabel("Operation Num")
    # labels = ["[128,512]", "[128,768]",
    #           "[256,512]", "[256,768]",
    #           "[512,512]", "[512,768]"]
    # # plt2.set_xticks(labels)
    # plt2.legend()
    #
    # plt.xticks(x, labels)
    # fig.suptitle("MAC Operation")
    # plt.show()

    print(y_mac)
    return 0

def main_1():
    x = np.array([1, 2, 3, 4, 5 ,6,7,8,9,10])

    plt.plot(x, x**2, color='red')


    plt.show()





if __name__ == "__main__":
    main_1()
