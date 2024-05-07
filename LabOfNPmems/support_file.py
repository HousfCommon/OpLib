import os
import random as r

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from evaluation import correlation_coefficient, compute_RMSE, accuracy, \
    fit_performance, set_class
from init_project import create_dirs, set_seeds, args_set

def data_process(path, tag):
    tar_df = pd.read_csv(os.path.join(path, tag), header=None)
    times = list(map(int, set(tar_df.values[:, -1])))



    last_columns = tar_df[tar_df.columns[-1]]

    c_max = np.max(tar_df.values[:, :-1])

    # targets

    row_targets = [np.float32(tar_df[last_columns == time].iloc[:, :-1].values/c_max)
                   for time in times][299]
    return row_targets


def evaluation_fit_figure(y_hot, y_cold, save_dir):
    # random_extraction_exps = r.sample(list(range(len(self.test_exps))),
    #                                            self.test_exps_num)

    # for i in range(len(y_pred_list)):
    #     y_pred = np.stack(y_pred_list[i]).reshape(-1, 1)
    #     y_target = np.stack(y_target_list[i]).reshape(-1, 1)
    #     w, line = fit_performance(y_pred, y_target)
    #     print(i, w)


    fig = plt.figure(figsize=(6.5,6))
    # fig, axes = plt.subplots(nrows=4, ncols=2)
    # for ax, i in zip(axes.flatten(), list(range(4))):
    y_hot = np.array(y_hot).reshape(-1, 1)
    y_cold = np.array(y_cold).reshape(-1, 1)
    w, line = fit_performance(y_hot, y_cold)
    plt.scatter(y_hot, y_cold, label='Adjusted data')
    plt.plot(y_hot, line, c='r', label='Fit: y=' + str(w) + 'x',linewidth=3.0)
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.xlabel(r'$\mathbf{c/c_{max}}$ (hot)',fontsize=17,weight='bold')

    plt.ylabel(r'$\mathbf{c/c_{max}}$ (cold)',fontsize=15,weight='bold')

    plt.xticks(fontsize=15,weight='bold')
    plt.yticks(fontsize=13,weight='bold')
    font = {'weight': 'bold', 'size': 16}
    plt.legend(fontsize=15,prop=font,loc=4)
    plt.title('t = 60 s',fontsize=20,weight='bold')
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    plt.savefig(os.path.join(save_dir, "fit_performance.eps"),dpi=1200)
    plt.savefig(os.path.join(save_dir, "fit_performance.tiff"), dpi=1200)
    plt.show()

def evaluation_mse(y_cold, y_hot, save_dir):

    pred_target_pairs = random_select(y_cold, y_hot)
    # compute error
    error = []
    for i in range(len(pred_target_pairs)):
        rmse = compute_RMSE(pred_target_pairs[0][:,i], pred_target_pairs[1][:,i])
        error.append(rmse)
    print(error)
    print(sum(error)/len(pred_target_pairs))

    # figure error in time:[0.120]s
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = np.arange(0, 120, 0.2)
    text = ['(a)', '(b)', '(c)', '(d)']

    for ax, i in zip(axes.flatten(), list(range(4))):
        ax.plot(t, pred_target_pairs[0][:,i],'.-', label='y_cold')
        ax.plot(t, pred_target_pairs[1][:,i], label='y_hot')
        ax.set_xlabel('time/s',fontsize=15)
        ax.set_ylabel('concentration (ppm)',fontsize=15)
        ax.set_title(text[i],fontsize=15)
        ax.xticks(fontsize=12)
        ax.yticks(fontsize=12)

        ax.legend(fontsize=14)

    plt.savefig(os.path.join(save_dir, "mse_performance.png"),dpi=1200)
    plt.show()

def random_select(y_cold, y_hot):
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

    extract_num = 4
    # the be choosed exp's y_pred
    y_cold = np.array(y_cold).squeeze()
    # the be choosed exp's y_target
    y_hot = np.array(y_hot).squeeze()
    points_num = y_cold.shape[1]

    # random choice 5 points in target points
    random_extraction_points = r.sample(list(range(points_num)), extract_num)
    # random_extraction_points = [112, 114, 728, 1043]
    # random_extraction_points = [112, 113, 728, 1043]
    print(random_extraction_points)

    y_cold_points = y_cold[:, random_extraction_points]
    y_hot_points = y_hot[:, random_extraction_points]


    return (y_cold_points, y_hot_points)

set_seeds(seed=1234, cuda=True)
root = '/media/myharddisk/data/supporting_data'
# random_extraction = r.sample(list(range(1053)), 4)
# random_extraction = list(range(0,1053))
# print(random_extraction)
y_hot = data_process(root,'c_tar_hot.csv')
y_cold = data_process(root,'c_tar_cold.csv')
evaluation_fit_figure(y_hot, y_cold, root)

# evaluation_mse(y_cold, y_hot, root)