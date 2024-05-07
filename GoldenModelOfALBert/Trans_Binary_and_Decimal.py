#--just practice--
from decimal import Decimal
import torch
import numpy as np
import math


def bTod(n, pre=4):
    '''
    把一个带小数的二进制数n转换成十进制
    小数点后面保留pre位小数
    '''
    string_number1 = str(n) # number1 表示二进制数，number2表示十进制数
    decimal = 0  # 小数部分化成二进制后的值
    flag = False
    for i in string_number1: # 判断是否含小数部分
        if i == '.':
            flag = True
            break
    if flag: # 若二进制数含有小数部分
        string_integer, string_decimal = string_number1.split('.') # 分离整数部分和小数部分
        for i in range(len(string_decimal)):
            decimal += 2**(-i-1)*int(string_decimal[i])  # 小数部分化成二进制
        number2 = int(str(int(string_integer, 2))) + decimal
        return round(number2, pre)
    else: # 若二进制数只有整数部分
        return int(string_number1, 2) # 若只有整数部分 直接一行代码二进制转十进制 python还是骚


def dTob(n, pre=4):
    '''
    把一个带小数的十进制数n转换成二进制
    小数点后面保留pre位小数
    '''
    string_number1 = str(n) # number1 表示十进制数，number2表示二进制数
    flag = False
    for i in string_number1: # 判断是否含小数部分
        if i == '.':
            flag = True
            break
    if flag:
        string_integer, string_decimal = string_number1.split('.') # 分离整数部分和小数部分
        integer = int(string_integer)
        decimal = Decimal(str(n)) - integer
        l1 = [0,1]
        l2 = []
        decimal_convert = ""
        while True:
           if integer == 0: break
           x,y = divmod(integer, 2)  # x为商，y为余数
           l2.append(y)
           integer = x
        string_integer = ''.join([str(j) for j in l2[::-1]])  # 整数部分转换成二进制
        i = 0
        while decimal != 0 and i < pre:
            result = int(decimal * 2)
            decimal = decimal * 2 - result
            decimal_convert = decimal_convert + str(result)
            i = i + 1
        string_number2 = string_integer + '.' + decimal_convert
        return float(string_number2)
    else: # 若十进制只有整数部分
        l1 = [0,1]
        l2 = []
        while True:
           if n == 0: break
           x,y = divmod(n, 2)  # x为商，y为余数
           l2.append(y)
           n = x
        string_number = ''.join([str(j) for j in l2[::-1]])
        return int(string_number)


def Transform_D_To_B(input, int_bit_width=4, tol_bit_width=8):
    input_r = input

    if torch.is_tensor(input):
        input = input.numpy()

    fra_bit_width = tol_bit_width - int_bit_width
    precision = 2**(-fra_bit_width)
    mask = input > 0
    output = np.array(input/precision, dtype=int)*mask * precision \
             + (np.array(input/precision, dtype=int))*~mask * precision

    if torch.is_tensor(input_r):
        output = torch.tensor(output)

    return output.float()


def Del_Frac(input, int_bit_width=4, tol_bit_width=8):

    fra_bit_width = tol_bit_width - int_bit_width
    output = input * 2**fra_bit_width

    return output


def Get_BitWidth_of_Integer(
        input
        , name
):
    max_ceil_input = torch.max(torch.ceil(input))
    min_floor_input = torch.min(torch.floor(input))

    int_bw = torch.log(torch.max(torch.abs(min_floor_input), max_ceil_input)).numpy() / math.log(2) + 1
    # print("range of %s is (%d , %d) " % (name, min_floor_input.numpy(), max_ceil_input.numpy()),
    #       "Bitwidth of Integer is %d " % int_bw)

    if np.abs(int_bw - int(int_bw)) >= 0.5 and int_bw > 0:
        int_bw = int(int_bw) + 1
    elif np.abs(int_bw - int(int_bw)) >= 0.5 and int_bw < 0:
        int_bw = int(int_bw) - 1
    else:
        int_bw = int(int_bw)

    return int_bw


def Get_BitWidth_of_Decimal(input, name, tol_bit_width=8):
    max_floor_input = torch.max(torch.floor(input))
    min_ceil_input = torch.min(torch.ceil(input))
    input_r = torch.max(torch.abs(input))
    # print("input_r", input_r)
    i = 0
    int_bit_width = Get_BitWidth_of_Integer(input, name)
    # print(int_bit_width)

    if max_floor_input == 0 and max_floor_input == min_ceil_input:
        while input_r.numpy() < 0:
            input_r *= 2
            i += 1

        fra_bit_width = tol_bit_width - 1 + i - int_bit_width
    else:
        fra_bit_width = tol_bit_width - int_bit_width

    print("range of %s is (%f , %f) " % (name, torch.min(input).numpy(), torch.max(input).numpy()),
          "Bitwidth of Decimal is %d " % fra_bit_width)

    return fra_bit_width


def main():
    x = np.random.randn(4, 4)
    torch.set_printoptions(precision=8)
    precision = 2**(-7)
    # print("precision is ", precision, precision*(2**7-1))
    mask = x > 0
    y = np.array(x/precision, dtype=int)*mask * precision + (np.array(x/precision, dtype=int) + 1)*~mask * precision

    # decimal_abs_y = np.abs((y - np.floor(y) - ~mask))
    # integer_abs_y = np.abs(np.floor(y))

    x_tensor = torch.rand(4, 4)
    fra_bw = Get_BitWidth_of_Decimal(input=x_tensor, name='testcase_x')
    y_tensor = Transform_D_To_B(x_tensor, int_bit_width=8-fra_bw, tol_bit_width=8)

    z = Del_Frac(y_tensor, int_bit_width=8-fra_bw, tol_bit_width=8)


    print(y)
    # print(floor_y)
    # print(del_y)
    # print(abs_y)
    print(z)
    return 0


def main_0():
    Basepath = '/Users/huanghuangtao/Desktop/'
    MODELNAME = '../../NLPOFLAB/checkpoint-79300/'
    NameBase = "albert.encoder.albert_layer_groups.0.albert_layers.0.attention."
    model = torch.load(MODELNAME + 'pytorch_model.bin', map_location=torch.device('cpu'))
    torch.set_printoptions(precision=8)

    for key in {"query.weight",
                "query.bias",
                "key.weight",
                "key.bias",
                "value.weight", "value.bias", "dense.weight", "dense.bias",
                "LayerNorm.weight",
                "LayerNorm.bias"
                }:
        x = model[NameBase+key]
        tol_bw = 8
        fra_bw = Get_BitWidth_of_Decimal(input=x, name=key)
        print(fra_bw)

        output = Transform_D_To_B(input=x, int_bit_width=tol_bw-fra_bw, tol_bit_width=tol_bw)
        output = Del_Frac(input=output, int_bit_width=tol_bw-fra_bw, tol_bit_width=tol_bw)
        print(output)



if __name__ == '__main__':
    main()
