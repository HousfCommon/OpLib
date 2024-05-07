#--just practice--
import numpy as np


def main():
    input_data = np.random.randn(1, 1, 8, 4)

    N, C, H, W = input_data.shape
    filter_w, filter_h = (4,1)
    stride_w, stride_h = (4,1)

    out_h = (H - filter_h) // stride_h + 1
    out_w = (W - filter_w) // stride_w + 1
    print("out", out_h, out_w)

    img = input_data
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):

        y_max = y + stride_h*out_h
        print("y",y,y_max)
        for x in range(filter_w):
            x_max = x + stride_w*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride_h, x:x_max:stride_w]

    col1 = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    print(img)
    print(col)
    print(col1)
    # img2 = np.random.randn(4,5)
    # print(img2)
    # print(img2[0:3:2,0:3:2])
    # print(img2[0:3:2,1:4:2])


if __name__ == "__main__":
    main()
