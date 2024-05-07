#--just practice--
import numpy as np
import torch


def main():
    a = torch.tensor(np.random.randn(10,10))

    print(a.shape)


if __name__ == '__main__':
    main()