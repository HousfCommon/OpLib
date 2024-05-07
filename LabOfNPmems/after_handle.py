import os
import random as r
import collections

import numpy as np
import torch
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd

from init_project import set_seeds,args_set
from evaluation import correlation_coefficient, compute_RMSE, accuracy, \
    fit_performance, set_class


def random_select_data(random_exp, random_extraction_points, y):
    '''
    y_pred/target_points = [
    1 time: point1 poin2 poin3 point4
    2 time: point1 poin2 poin3 point4
    ...
    600time: point1 poin2 poin3 point4

    pred_target_pairs = [
    [y_pred_points:[1time, 2time, ... 600time],
    y_target_points:[1time, 2time, ... 600time]
    ],
    ...
    ]
    '''
    y_exp = np.stack(y[random_exp])

    y_points = list(y_exp[0][:, random_extraction_points])
    for j in range(1, len(y_exp)):
        y_points.append(list(y_exp[j][-1, random_extraction_points]))

    return y_points


def evaluation_mse_figure(model_target, model_result, save_dir):

    t = np.arange(2, 120, 0.2)
    fig = plt.figure(figsize=(9.2,7))
    # ax = fig.add_subplot(2, 1, 1)

    # figure的百分比, 从figure 10%的位置开始绘制, 宽高是figure的80%
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    # 获得绘制的句柄
    ax = fig.add_axes([left, bottom, width, height])

    ax.plot(t, model_target, '-v', label='Target data',linewidth=3.0)
    for key, value in model_result.items():
        ax.plot(t, value, '-', label=key,linewidth=3.0)

    ax.set_xlabel('time (s)',fontsize=22,weight='bold')
    ax.set_ylabel('concentration (ppm)',fontsize=22,weight='bold')
    plt.xticks(fontsize=18, weight='bold')
    plt.yticks(fontsize=18, weight='bold')
    font = {'weight': 'bold', 'size': 16}
    plt.legend(prop=font)
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    # ax3 = fig.add_subplot(2,1, 2)
    # ax3 = fig.add_axes([0.3,0.6,0.3,0.3])
    # ax3.plot(t, model_target, '-v', label='Target data',linewidth=3.0)
    # for key, value in model_result.items():
    #     ax3.plot(t, value, '-', label=key,linewidth=3.0)
    # ax3.set_xlim(4, 14)
    # ax3.set_ylim(6, 17)
    # plt.xticks(fontsize=18, weight='bold')
    # plt.yticks(fontsize=18, weight='bold')

    # ax4 = plt.gca()  # 获得坐标轴的句柄
    # ax4.spines['bottom'].set_linewidth(2)
    # ax4.spines['left'].set_linewidth(2)
    # ax4.spines['right'].set_linewidth(2)
    # ax4.spines['top'].set_linewidth(2)

    plt.savefig(os.path.join(save_dir, "test.eps"),dpi=1200)
    plt.show()

def space_concentration_figure(y_pred_list, y_tar_list, save_dir):
    # choose a exp as example
    random_exp = 9
    print(random_exp)
    # the be choosed exp's y_pred
    choose_y_pred_exp = np.stack(y_pred_list[random_exp])
    # the be choosed exp's y_target
    choose_y_target_exp = np.stack(y_tar_list[random_exp])
    seq_num = choose_y_pred_exp.shape[0]

    # random choice 5 points in target points
    # random_extraction_seq = r.sample(list(range(seq_num)), 1)
    random_extraction_seq = 136
    print(random_extraction_seq)

    y_pred_seq = y_change(choose_y_pred_exp, random_extraction_seq)
    y_tar_seq = y_change(choose_y_target_exp, random_extraction_seq)
    torch.save(y_pred_seq, os.path.join(save_dir, 'y_pred_seq'))
    torch.save(y_tar_seq, os.path.join(save_dir, 'y_tar_seq'))

def space_concentration_figure_(y_pred_list, y_tar_list, save_dir):
    '''
    找到准确率最大的一段时间序列
    :param y_pred_list:
    :param y_tar_list:
    :return:
    '''
    # choose a exp as example
    random_exp = 9
    print(random_exp)
    # the be choosed exp's y_pred
    y_pred_exp = np.stack(y_pred_list[random_exp])
    # the be choosed exp's y_target
    y_target_exp = np.stack(y_tar_list[random_exp])
    seq_num = y_pred_exp.shape[0]
    out = {}

    for i in range(seq_num):
        y_pred_seq = y_change(y_pred_exp, i)
        y_tar_seq = y_change(y_target_exp, i)

        y_pred_class = np.array(
            list(map(set_class, np.array(y_pred_seq).reshape(-1))))
        y_tar_class = np.array(
            list(map(set_class, np.array(y_tar_seq).reshape(-1))))
        acc = accuracy(y_pred_class, y_tar_class)
        out[i] = acc

    print(sorted(zip(out.values(), out.keys())),
          file=open(os.path.join(save_dir, 'sequence_accuracy.txt'),'w'))


def y_change(y_exp, random_extraction_seq, pred_seq_len=5):
    '''
    map data in real space
    '''
    y_out = []

    for i in range(pred_seq_len):
        y_time = y_exp[random_extraction_seq, i, :]
        y_insert_time = np.insert(y_time, 263, 0)
        y_insert_time = np.insert(y_insert_time, 527, 0)
        y_insert_time = np.insert(y_insert_time, 791, 0)
        # y_change = y_insert_time.reshape(24, 11, -1)
        x = 0
        y_change = np.zeros([24, 11, 4])
        for z in list(range(4)):
            for y in list(range(11)):
                y_change[:,y,z] = y_insert_time[x: x+24]
                x += 24

        y_out.append(y_change)
    return y_out

def multi_model_compare_figure(save_dir):
    model_result = {}
    result_path = {}
    result_path['Baseline1'] = '/media/myharddisk/result/TEMPORAL_ONLY_NEW'
    result_path['Baseline2'] = '/media/myharddisk/result/CNN_ONLY_NEW'
    result_path['Baseline3'] = '/media/myharddisk/result/MQ_MLP'
    result_path['Baseline4'] = '/media/myharddisk/result/CNN_SEQ2SEQ'
    result_path['MSSTP-SA net'] = '/media/myharddisk/result/pred_seq_len_1s'

    y_tar_list = torch.load(
        os.path.join(result_path['MSSTP-SA net'], 'y_tar_list'))

    test_exp_len = len(y_tar_list)
    # choose a exp as example
    # random_exp = r.sample(list(range(test_exp_len)), 1)[0]
    random_exp =196
    # random choice 1 points in target points
    # random_extraction_points = r.sample(list(range(1053)), 1)
    random_extraction_points = [1048]
    print(random_extraction_points)
    model_target = random_select_data(random_exp,
                                      random_extraction_points,
                                      y_tar_list)

    for key, item in result_path.items():
        y_pred_list_path = os.path.join(result_path[key], 'y_pred_list')
        y_pred_list = torch.load(y_pred_list_path)
        model_result[key] = random_select_data(random_exp,
                                               random_extraction_points,
                                               y_pred_list)

    evaluation_mse_figure(model_target, model_result, save_dir)


    print('test')

def evaluation_fit_figure(y_pred_list, y_target_list,i, save_dir):
    # random_extraction_exps = r.sample(list(range(len(self.test_exps))),
    #                                            self.test_exps_num)

    # for i in range(len(y_pred_list)):
    #     y_pred = np.stack(y_pred_list[i]).reshape(-1, 1)
    #     y_target = np.stack(y_target_list[i]).reshape(-1, 1)
    #     w, line = fit_performance(y_pred, y_target)
    #     print(i, w)

    # test_list = [7,9,16,39]

    # for i in range(4):
    #     y_pred = np.stack(y_pred_list[i]).reshape(-1, 1)
    #     y_target = np.stack(y_target_list[i]).reshape(-1, 1)
    #     w, line = fit_performance(y_pred, y_target)
    #     fig = plt.figure(figsize=(6,5))
    #     plt.scatter(y_pred, y_target, label='Adjusted data')
    #     plt.plot(y_pred, line, c='r', label='Fit: y=' + str(w) + 'x',linewidth=3.0)
    # # ax.set_xticks(fontsize=13)
    #     # ax.set_yticks(fontsize=13)
    #     plt.xlabel('Model output',fontsize=15,weight='bold')
    #     plt.ylabel('CFD output',fontsize=15,weight='bold')
    #     # plt.set_title(text[i],fontsize=15)
    #     font = {'weight': 'bold', 'size': 16}
    #     plt.legend(fontsize=15,prop=font)
    #     plt.xticks(fontsize=12,weight='bold')
    #     plt.yticks(fontsize=12,weight='bold')
    #     ax = plt.gca()  # 获得坐标轴的句柄
    #     ax.spines['bottom'].set_linewidth(2)
    #     ax.spines['left'].set_linewidth(2)
    #     ax.spines['right'].set_linewidth(2)
    #     ax.spines['top'].set_linewidth(2)
    #     plt.savefig(os.path.join(save_dir,str(i)+"fit.png"))

    y_pred = np.stack(y_pred_list).reshape(-1, 1)
    y_target = np.stack(y_target_list).reshape(-1, 1)
    w, line = fit_performance(y_pred, y_target)
    fig = plt.figure(figsize=(7,6))
    plt.scatter(y_pred, y_target, label='Adjusted data')
    plt.plot(y_pred, line, c='r', label='Fit: y=' + str(w) + 'x',linewidth=3.0)
    # ax.set_xticks(fontsize=13)
        # ax.set_yticks(fontsize=13)
    plt.xlabel('Model output',fontsize=15,weight='bold')
    plt.ylabel('CFD output',fontsize=14,weight='bold')
        # plt.set_title(text[i],fontsize=15)
    font = {'weight': 'bold', 'size': 16}
    plt.legend(fontsize=15,prop=font,loc='upper left')
    plt.xticks(fontsize=12,weight='bold')
    plt.yticks(fontsize=10,weight='bold')
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.savefig(os.path.join(save_dir,str(i)+"fit.png"),dpi=600)

    plt.show()

def evaluation_index(y_pred, y_tar):
    y_pred_cat = y_pred[0].reshape(-1)
    y_target_cat = y_tar[0].reshape(-1)
    for j in range(1, len(y_pred)):
        y_pred_cat = np.concatenate((y_pred_cat, y_pred[j].reshape(-1)))
        y_target_cat = np.concatenate((y_target_cat, y_tar[j].reshape(-1)))

    rmse, rela_error_mean = compute_RMSE(y_pred_cat, y_target_cat)
    r = correlation_coefficient(y_pred_cat, y_target_cat)
    accuracy_rate = accuracy(y_pred_cat, y_target_cat)

    return rmse, rela_error_mean, r, accuracy_rate

def evaluation_mse(y_pred_list, y_target_list,exp, points, save_dir):

    pred_target_pairs = random_select(y_pred_list, y_target_list,exp, points)
    # compute error
    rmse_list = []
    r_error_max_list = []
    r_error_mean_list = []
    for i in range(len(pred_target_pairs)):
        rmse,r_error_max,r_error_mean = compute_RMSE(pred_target_pairs[i][0], pred_target_pairs[i][1])
        rmse_list.append(rmse)
        r_error_max_list.append(r_error_max)
        r_error_mean_list.append(r_error_mean)
    print(r_error_mean_list)


    # figure error in time:[0.120]s
    # fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    t = np.arange(0, 118, 0.2)


    for i in range(2):
        fig = plt.figure(figsize=(6,6))
        plt.plot(t, pred_target_pairs[i][0],'.-', label='Prediction',linewidth=3.0)
        plt.plot(t, pred_target_pairs[i][1], label='Target data',linewidth=3.0)
        plt.xlabel('time (s)',fontsize=18,weight='bold')
        plt.ylabel('concentration (ppm)',fontsize=15.5,weight='bold')
        plt.xticks(fontsize=15,weight='bold')
        plt.yticks(fontsize=12.5,weight='bold')
        font = {'weight': 'bold', 'size': 14}
        plt.legend(prop=font)
        ax = plt.gca()  # 获得坐标轴的句柄
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        plt.grid(linestyle='--')
        # ax.set_title(text[i],fontsize=16)
        plt.savefig(os.path.join(save_dir, str(exp)+str(i)+"_mse_performance.pdf"), dpi=1200)

    # plt.savefig(os.path.join(save_dir, "mse_performance.png"),dpi=1200)
    plt.show()

def random_select(y_pred_list, y_target_list, random_exp, random_extraction_points):
    '''
    y_pred/target_points = [
    1 time: point1 poin2 poin3 point4
    2 time: point1 poin2 poin3 point4
    ...
    600time: point1 poin2 poin3 point4

    pred_target_pairs = [
    [y_pred_points:[1time, 2time, ... 600time],
    y_target_points:[1time, 2time, ... 600time]
    ],
    ...
    ]
    '''
     # choose a exp as example

    # random_exp = 9
    extract_num = 2
    # the be choosed exp's y_pred
    y_pred_exp = np.stack(y_pred_list[random_exp])
    # the be choosed exp's y_target
    y_target_exp = np.stack(y_target_list[random_exp])
    points_num = y_pred_exp.shape[2]

    # random choice 5 points in target points
    # random_extraction_points = r.sample(list(range(points_num)), 2)
    # random_extraction_points = [112, 114, 728, 1043]
    # random_extraction_points = [1047,1046]
    print(random_extraction_points)

    y_pred_points = y_pred_exp[0][:, random_extraction_points]
    y_target_points = y_target_exp[0][:, random_extraction_points]
    for j in range(1, len(y_pred_exp)):
        y_pred_points = np.concatenate((y_pred_points,
            y_pred_exp[j][-1, random_extraction_points].reshape(1,-1)))
        y_target_points = np.concatenate((
            y_target_points,
            y_target_exp[j][-1, random_extraction_points].reshape(1,-1)))

    pred_target_pairs = [[y_pred_points[:, i], y_target_points[:, i]]
                             for i in range(extract_num)]

    return pred_target_pairs


def main():
    args = args_set('big')
    print('test123')
    set_seeds(seed=1234, cuda=False)
    # path = '/media/myharddisk/result/pred_seq_len_1s'
    path = '/media/myharddisk/result/pred_seq_len_5s_new'
    # path = '/media/myharddisk/copy/MSSTP_SA_net/result/change_loss'
    y_tar_list = torch.load(os.path.join(path, 'y_tar_list'))
    y_pred_list = torch.load(os.path.join(path, 'y_pred_list'))
    # multi_model_compare_figure(path)
    # space_concentration_figure(y_pred_list, y_tar_list, args.save_dir)
# -------------mse------------------------------------------------------
#     d = {0:[541,543], 21: [196,1], 38: [1046,1047]}
#     d = {38: [1046,1047]}
#     for exp,nums in d.items():
#         evaluation_mse(y_pred_list, y_tar_list, exp, nums, os.path.join(path,'mse_fig'))


    exp_index = []
    test_exp = np.load('exp_list.npy')
    print('index','exp','r','rmse','acc','error_max','error_mean')
    for i in range(len(y_tar_list)):
        # evaluation_fit_figure(y_pred_list[i], y_tar_list[i], i,
        #                       save_dir=os.path.join(path,'fit_fig'))
        rmse, error_mean, r, acc =evaluation_index(y_pred_list[i], y_tar_list[i])
        exp_index.append([test_exp[i],r, rmse, acc, error_mean])
        print(i, test_exp[i], r, rmse, acc, error_mean)

    exp_index_csv = pd.DataFrame(exp_index)
    # exp_index_csv.head = ['index','exp','r','rmse','acc','error']
    exp_index_csv.to_csv(os.path.join(path,'new_test_index.csv'))



if __name__ == '__main__':
    main()