#--just practice--
import numpy as np
import matplotlib.pyplot as plt


def main():
    len = 4
    max = 2
    x = np.linspace(0, max, len)
    print(x)
    y = np.sqrt(x)
    print(y)
    k = np.zeros(len-1)
    for i in range(len-1):
        k[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
    print(k)
    x_1 = np.linspace(0, max+0.05, len*1000)
    y_1 = np.sqrt(x_1)

    # k = [1.0623, 0.5069, 0.3896]
    # b = [0.1795, 0.4879, 0.6393]
    # y_2 = np.zeros(len)
    # for j in range(len-1):
    #     y_2[j+1] = k[j]*x[j+1]+b[j]
    #
    # print(y_2)

    plt.figure(figsize=(16, 7))

    plt.subplot(1,2,1)
    plt.plot(x, y, label='Continuous Approximation', color='darkseagreen')
    plt.plot(x_1, y_1, label="Original", linestyle='--', color='navy')

    alpha = 0
    for i in range(len):
        # plt.axvline(x[i], ymin=0, ymax=y[i], ls=':', c='black')
        # plt.axhline(y[i], xmin=0, xmax=x[i]/2, linestyle=':', c='black')
        plt.plot([x[i], x[i]],[0, y[i]], linestyle=':', c='black')
        plt.plot([0, x[i]], [y[i], y[i]], linestyle=':', c='black')

    # x_2 = np.linspace(0, max, len*2)
    # plt.xticks(x)
    plt.legend(loc='upper left')
    plt.axis([0, max+0.1, 0, np.max(y)+0.1])
    plt.xlabel("x")
    plt.ylabel("sqrt(x)")

    plt.subplot(1, 2, 2)
    k = [1.0623, 0.5069, 0.3896]
    b = [0.1795, 0.4879, 0.6393]
    y_2 = np.zeros((len - 1) * 2)
    for j in range(len - 1):
        y_2[2 * j] = k[j] * x[j] + b[j]
        y_2[2 * j + 1] = k[j] * x[j + 1] + b[j]
        if j==0:
            plt.plot([x[j], x[j + 1]], [y_2[2 * j], y_2[2 * j + 1]], label='Discontinuous Approximation', color='orange')
        else:
            plt.plot([x[j], x[j + 1]], [y_2[2 * j], y_2[2 * j + 1]], color='orange')
    plt.plot(x_1, y_1, label="Original", linestyle='--', color='navy')

    for i in range(len):
        plt.axvline(x[i], ls=':', c='black')
        # plt.axhline(y[i], xmin=0, xmax=x[i]/2, linestyle=':', c='black')

    # x_2 = np.linspace(0, max, len*2)
    # plt.xticks(x)
    plt.legend(loc='upper left')
    plt.axis([0, max+0.1, 0, np.max(y)+0.1])
    plt.xlabel("x")
    plt.ylabel("sqrt(x)")

    plt.show()


def gelu(x):
    return 0.5*x*(1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))


def main_2():

    plt.figure(figsize=(24, 6))
    plt.subplot(1, 3, 1)
    len = 8
    max = 0
    min = -7
    x = np.linspace(min, max, len)
    print(x)
    y = np.exp(x)
    x_1 = np.linspace(min, max, len * 1000)
    y_1 = np.exp(x_1)

    k = [0, 0.0042, 0.0114, 0.0310, 0.0844, 0.2293, 0.6233]
    b = [0, 0.0274, 0.0630, 0.1402, 0.2968, 0.5775, 0.9464]
    y_2 = np.zeros((len - 1) * 2)
    for j in range(len - 1):
        y_2[2 * j] = k[j] * x[j] + b[j]
        y_2[2 * j + 1] = k[j] * x[j + 1] + b[j]
        if j == 0:
            plt.plot([x[j], x[j + 1]], [y_2[2 * j], y_2[2 * j + 1]], label='Discontinuous Approximation',
                     color='crimson')
        else:
            plt.plot([x[j], x[j + 1]], [y_2[2 * j], y_2[2 * j + 1]], color='crimson')
    plt.plot(x_1, y_1, label="Original", linestyle='--', color='navy')

    for i in range(len):
        plt.axvline(x[i], ls=':', c='black')

    plt.legend(loc='upper left')
    plt.axis([min-0.1, 0.1, 0, np.max(y) + 0.1])
    plt.xlabel("x")
    plt.ylabel("exp(x)")


    plt.subplot(1, 3, 2)

    len = 11
    max = 2.4
    min = 0
    x = np.linspace(min, max, len)
    print(x)
    y = np.sqrt(x)
    x_1 = np.linspace(min, max, len * 1000)
    y_1 = np.sqrt(x_1)

    k = [2.1162, 0.9294, 0.7102, 0.5989, 0.5277, 0.4772, 0.4388, 0.4084, 0.3836, 0.3628]
    b = [0.0632, 0.2665, 0.3502, 0.4163, 0.4730, 0.5234, 0.5693, 0.6117, 0.6514, 0.6888]
    y_2 = np.zeros((len - 1) * 2)
    for j in range(len - 1):
        y_2[2 * j] = k[j] * x[j] + b[j]
        y_2[2 * j + 1] = k[j] * x[j + 1] + b[j]
        if j == 0:
            plt.plot([x[j], x[j + 1]], [y_2[2 * j], y_2[2 * j + 1]], label='Discontinuous Approximation',
                     color='darkseagreen')
        else:
            plt.plot([x[j], x[j + 1]], [y_2[2 * j], y_2[2 * j + 1]], color='darkseagreen')
    plt.plot(x_1, y_1, label="Original", linestyle='--', color='navy')

    for i in range(len):
        plt.axvline(x[i], ls=':', c='black')

    plt.legend(loc='upper left')
    plt.axis([0, max + 0.1, 0, np.max(y) + 0.1])
    plt.xlabel("x")
    plt.ylabel("sqrt(x)")


    plt.subplot(1, 3, 3)

    len = 7
    max = 3
    min = -3
    x = np.linspace(min, max, len)
    print(x)
    y = gelu(x)
    x_1 = np.linspace(min, max, len * 1000)
    y_1 = gelu(x_1)

    k = [-0.0405, -0.1183, 0.1500, 0.8500, 1.1183, 1.0405]
    b = [-0.1198, -0.2788, -0.0515, -0.0515, -0.2788, -0.1198]
    y_2 = np.zeros((len - 1) * 2)
    for j in range(len - 1):
        y_2[2 * j] = k[j] * x[j] + b[j]
        y_2[2 * j + 1] = k[j] * x[j + 1] + b[j]
        if j == 0:
            plt.plot([x[j], x[j + 1]], [y_2[2 * j], y_2[2 * j + 1]], label='Discontinuous Approximation',
                     color='orange')
        else:
            plt.plot([x[j], x[j + 1]], [y_2[2 * j], y_2[2 * j + 1]], color='orange')
    plt.plot(x_1, y_1, label="Original", linestyle='--', color='navy')

    for i in range(len):
        plt.axvline(x[i], ls=':', c='black')

    plt.legend(loc='upper left')
    plt.axis([min-0.1, max + 0.1, np.min(y)-0.1, np.max(y) + 0.1])
    plt.xlabel("x")
    plt.ylabel("gelu(x)")
    plt.show()


if __name__=="__main__":
    main_2()

