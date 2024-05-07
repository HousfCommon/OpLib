import os
import torch
import numpy as np
from argparse import Namespace

def create_dirs(dirpath):#创建目录函数
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Set Numpy and PyTorch seeds
def set_seeds(seed, cuda):#设置种子函数
    np.random.seed(seed)#用于生成指定随机数
    torch.manual_seed(seed)#随机化初始种子
    if cuda:
        torch.cuda.manual_seed_all(seed)#为所有的GPU设置种子

def args_set(pattern):#解析输入参数
    """ Parse input argument"""#解析输入参数
    if pattern == 'small':#命名空间
        args = Namespace(
            seed=1234,#生成伪随机数时指定的随机抽样起点，只要设置的Seed是一样的，就保证得到同样的一组随机模拟数据
            cuda=False,
            shuffle=True,#把数组中的元素按随机顺序随机排列
            data_folder="E:/0xjw_learning/home/data/processed_data_test_1g",#处理好之后的数据保存的文件夹
            spatial_model_state_file="spatial_model.pth",#空间模型文件指定的路径：.pth文件会书写一些路径，一行一个
            time_model_state_file='time_model.pth',#时间模型文件指定的路径
            save_dir="E:/0xjw_learning/home/data/result/checkpoint_test",#保存测试目录
            save_sample_path='E:/0xjw_learning/home/data/sample_test1',#保存样本路径
            input_seq_len=10,#数据长度
            pred_seq_len=15,#时间长度
            train_size=0.2,
            val_size=0.2,
            test_size=0.2,
            alpha=0.8,
            num_epochs=1,
            extract_num=4,
            early_stopping_criteria=5,
            learning_rate=1e-3,
            batch_size=32,
            dropout_p=0.5,
            teacher_forcing_ratio=0.5,
            resume=0
        )
    elif pattern == 'big':
        args = Namespace(
            seed=1234,#设置种子函数
            cuda=False,
            shuffle=True,#把数组中的元素按随机序列排列
            data_folder="F:/0xjw_learning/home/data/processed_data_test_5g",
            spatial_model_state_file="spatial_model.pth",
            time_model_state_file='time_model.pth',
            # save_dir="/media/myharddisk/result/pred_seq_len_1s",
            # save_sample_path='/home/zhyhou/xjw/sample',
            save_dir="F:/0xjw_learning/home/data/result/pred_seq_len_5s_new",#保存测试目录
            save_sample_path='F:/0xjw_learning/home/data/sample_test',#保存样本路径
            # save_sample_path='/home/zhyhou/xjw/sample',
            # save_dir="/media/myharddisk/result/pred_seq_len_7s",
            # save_sample_path='/home/zhyhou/xjw/data/sample_7s',
            input_seq_len=10,
            pred_seq_len=25,
            # pred_seq_len=45,
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            alpha=0,
            num_epochs=50,
            extract_num=4,
            early_stopping_criteria=5,
            learning_rate=1e-3,
            batch_size=32,
            dropout_p=0.5,
            teacher_forcing_ratio=0.5,
            resume=0
        )
    else:
        raise KeyError(pattern)
    return args

def args_train_state(early_stopping_criteria, learning_rate):#训练
    train_state = {
    'done_training': False,
    'stop_early': False,
    'early_stopping_step': 0,
    'early_stopping_best_val': 1e8,
    'early_stopping_criteria': early_stopping_criteria,
    'learning_rate': learning_rate,
    'epoch_index': 0,
    'train_loss': [],
    'val_loss': [],
    }
    return train_state