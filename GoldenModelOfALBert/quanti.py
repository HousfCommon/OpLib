import copy
import math
import numpy as np
from scipy import stats
import torch

QUANTIZE_NUM = 127
QUANTIZE_NUM_16 = 2**15-1
QUANTIZE_WINOGRAND_NUM = 31
STATISTIC = 1
INTERVAL_NUM = 2048


class QuantizeLayer:
    def __init__(self, name, blob_name, group_num):
        self.name = name
        self.blob_name = blob_name
        self.group_num = group_num
        self.weight_scale = np.zeros(group_num)
        self.blob_max = 0.0
        self.blob_distubution_interval = 0.0
        self.blob_distubution = np.zeros(INTERVAL_NUM)
        self.blob_threshold = 0
        self.blob_scale = 1.0
        self.group_zero = np.zeros(group_num)
        self.b_weight_scale = np.zeros(group_num)
        self.b_blob_scale = None

    def quantize_weight(self, weight_data, flag=False):
        # spilt the weight data by cout num
        blob_group_data = np.array_split(weight_data, self.group_num)
        for i, group_data in enumerate(blob_group_data):
            max_val = np.max(group_data)
            min_val = np.min(group_data)
            threshold = max(abs(max_val), abs(min_val))
            # if threshold < 0.0001:
            #     self.weight_scale[i] = 0
            #     self.group_zero[i] = 1
            # else:
            if(flag == True):
                self.weight_scale[i] = QUANTIZE_NUM_16 / threshold
            else:
                self.weight_scale[i] = QUANTIZE_NUM / threshold
            print("%-20s group : %-5d max_val : %-10f scale_val : %-10f" % (self.name + "_param0", i, threshold, self.weight_scale[i]))

    def initial_blob_max(self, blob_data):
        # get the max value of blob
        max_val = np.max(blob_data)
        min_val = np.min(blob_data)
        self.blob_max = max(self.blob_max, max(abs(max_val), abs(min_val)))

    def initial_blob_distubution_interval(self):
        self.blob_distubution_interval = STATISTIC * self.blob_max / INTERVAL_NUM
        print("%-20s max_val : %-10.8f distribution_intervals : %-10.8f" % (self.name, self.blob_max, self.blob_distubution_interval))

    def initial_histograms(self, blob_data):
        # collect histogram of every group channel blob
        th = self.blob_max
        hist, hist_edge = np.histogram(blob_data, bins=INTERVAL_NUM, range=(0, th))
        self.blob_distubution += hist

    def quantize_blob(self):
        # calculate threshold  
        distribution = np.array(self.blob_distubution)
        # pick threshold which minimizes KL divergence
        threshold_bin = threshold_distribution(distribution) 
        self.blob_threshold = threshold_bin
        threshold = (threshold_bin + 0.5) * self.blob_distubution_interval
        # get the activation calibration value
        self.blob_scale = QUANTIZE_NUM / threshold
        print("%-20s bin : %-8d threshold : %-10f interval : %-10f scale : %-10f" % (self.name, threshold_bin, threshold, self.blob_distubution_interval, self.blob_scale))

    def quantize_scale(self):
        self.b_blob_scale = math.log(self.blob_scale, 2)
        for index in range(self.group_num):
            self.b_weight_scale[index] = math.log(self.weight_scale[index], 2)
        print('data_scale: %d | weight_scale: %d' % (round(self.b_blob_scale), round(self.b_weight_scale[0])))


def weight_scale_gen(weight_blob, name, blob_name):
    if torch.is_tensor(weight_blob):
        weight_blob = weight_blob.numpy()
    quanitze_layer = QuantizeLayer(name, blob_name, 1)
    quanitze_layer.quantize_weight(weight_blob, False)
    k_weight = math.log(quanitze_layer.weight_scale, 2)
    w_scale = round(k_weight)

    return w_scale


def bias_scale_gen(bias_blob, name, blob_name):
    if torch.is_tensor(bias_blob):
        bias_blob = bias_blob.numpy()
    quanitze_layer = QuantizeLayer(name, blob_name, 1)
    quanitze_layer.quantize_weight(bias_blob, True)
    k_weight = math.log(quanitze_layer.weight_scale, 2)
    w_scale = round(k_weight)

    return w_scale


def data_scale_gen(blob_data, name, blob_name):
    if torch.is_tensor(blob_data):
        blob_data = blob_data.numpy()
    quanitze_layer = QuantizeLayer(name, blob_name, 1)
    quanitze_layer.initial_blob_max(blob_data)
    quanitze_layer.initial_blob_distubution_interval()
    quanitze_layer.initial_histograms(blob_data)
    quanitze_layer.quantize_blob()
    k_data = math.log(quanitze_layer.blob_scale, 2)
    data_scale = round(k_data)

    return data_scale


def quantize_data(din, scale):
    flag = 0
    if torch.is_tensor(din):
        din = din.numpy()
        flag = 1
    quan_res = np.round(din*(2**scale))
    quan_res = np.int32(quan_res)
    quan_res[quan_res > 127] = 127
    quan_res[quan_res < -128] = -128

    if flag:
        quan_res = torch.from_numpy(quan_res)

    return quan_res


def quantize_bias(din, scale):
    flag = 0
    if torch.is_tensor(din):
        din = din.numpy()
        flag = 1
    quan_res = np.round(din*(2**scale))
    quan_res = np.int32(quan_res)
    quan_res[quan_res > 2**15-1] = 2**15-1
    quan_res[quan_res < -2**15] = -2**15

    if flag:
        quan_res = torch.from_numpy(quan_res)

    return quan_res


def threshold_distribution(distribution, target_bin=128):
    """
    Return the best threshold value. 
    Ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    Args:
        distribution: list, activations has been processed by histogram and normalize,size is 2048
        target_bin: int, the num of bin that is used by quantize, Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL 
    """   
    distribution = distribution[1:]
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        p[threshold-1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        # 
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin
        
        # merge hist into num_quantized_bins bins
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()
        
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        
        # calculate kl_divergence between q and p
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value


def main():
    input_data = torch.randn(10, 10)
    in_weight_0 = torch.randn(10, 20)
    in_bias_0 = torch.randn(1, 20)
    in_weight_1 = torch.randn(10, 20)
    in_bias_1 = torch.randn(1, 20)

    x_scale = data_scale_gen(input_data, 'Linear0', 'Linear_input')
    w_scale = weight_scale_gen(in_weight_0, 'Linear0', 'Linear_weight')
    b_scale = bias_scale_gen(in_bias_0, 'Linear0', 'Linear_bias')

    x = quantize_data(input_data, x_scale)
    w = quantize_data(in_weight_0, w_scale)
    b = quantize_bias(in_bias_0, b_scale)


if __name__=='__main__':
    main()
    # weight_blob = np.array([[2, 0, 2], [0, 2, 0], [0, 0, 2]])
    # quanitze_layer = QuantizeLayer('wino_layer', 'wino_input', 1)
    # quanitze_layer.quantize_weight(weight_blob, False)
    # blob_data = np.array([[0, 2, 3, 4], [5, 6, 7, 8], [9, -10, 11, 12], [13, 14, 15, 11]])
    # quanitze_layer.initial_blob_max(blob_data)
    # quanitze_layer.initial_blob_distubution_interval()
    # quanitze_layer.initial_histograms(blob_data)
    # quanitze_layer.quantize_blob()
    # quanitze_layer.quantize_scale()








