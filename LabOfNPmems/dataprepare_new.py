import os
import math
import pickle
import pandas as pd
import numpy as np

from init_project import create_dirs, set_seeds, args_set

class DataPrepare(object):
    def __init__(self, save_dir, data_folder, save_sample_path, train_size,
                 val_size, test_size, input_seq_len, pred_seq_len, shuffle=True):
        self.save_dir = save_dir#保存测试目录
        self.data_folder = data_folder#数据文件夹
        self.save_sample_path = save_sample_path#保存样本路径
        self.train_size = train_size#训练尺寸
        self.val_size = val_size#监督尺寸
        self.test_size = test_size#测试尺寸
        self.input_seq_len = input_seq_len# 数据长度
        self.pred_seq_len = pred_seq_len#  时间长度
        self.shuffle = shuffle#把数组中的元素随机排列

        self.scales = []# ? ? ?

        create_dirs(self.save_sample_path) #保存样本路径--- F:\19 test of dataprepare\0xjw_learning\home\data\sample_5s
        #self.test_exps = np.load('F:/2 python练习/0xjw_learning/home/MSSTP_SA_net/test_exp_list.npy') # 测试清单

        self.train_exps, self.val_exps ,self.test_exps= self.exps_split()
        np.save('F:/0xjw_learning/home/MSSTP_SA_net/train_exp_list.npy', self.train_exps)# 训练清单
        np.save('F:/0xjw_learning/home/MSSTP_SA_net/val_exp_list.npy', self.val_exps)# 监督清单
        np.save('F:/0xjw_learning/home/MSSTP_SA_net/test_exp_list.npy',self.test_exps)#测试清单
        
        self.find_max()# ? ? ?
        np.save('scales.npy', self.scales)
        # print('test_exps: ', self.test_exps)
        # print('scales: ', self.scales)

    def create_data(self):#产生训练数据和监督数据
        split(data_folder=self.data_folder, exps=self.train_exps,# 通过指定分隔符对字符串进行切片
              input_seq_len=self.input_seq_len, pred_seq_len=self.pred_seq_len,
              save_sample_path=self.save_sample_path, tag='train',#产生训练文件夹
              scales=self.scales)

        split(data_folder=self.data_folder, exps=self.val_exps,# 通过指定分隔符对字符串进行切片
              input_seq_len=self.input_seq_len, pred_seq_len=self.pred_seq_len,
              save_sample_path=self.save_sample_path, tag='val',#产生监督文件夹
              scales=self.scales)

    def exps_split(self):#已掌握
        '''
        shuffle exp_list
        split input, space_target, time_target as train, val, test 将exp_list所有元素随机排序，将输入，空间和时间目标分为，训练，监督和测试

        '''
        exp_list = os.listdir(self.data_folder) # 返回指定的文件夹包含的文件或者文件夹的名字的列表
        np.save('E:/0xjw_learning/home/MSSTP_SA_net/exp_list.npy', exp_list)
        exp_length = len(exp_list)#获取exp_list的长度

        #for i in range(len(self.test_exps)):
            #exp_list.remove(self.test_exps[i])

        while '.DS_Store' in exp_list:# 目的在于存贮目录的自定义属性
            exp_list.remove('.DS_Store')
        if self.shuffle:
            np.random.shuffle(exp_list)# 随机排序exp_list

        # exp_length = len(exp_list)
        n_train = math.ceil(self.train_size * exp_length)  # 分出训练列表，并向上取整
        n_val = math.ceil(self.val_size * exp_length)#分出监督列表，并向上取整
        train_exps = exp_list[:n_train] #训练部分的数据集
        val_exps = exp_list[n_train:n_train + n_val] #监督部分的数据集
        test_exps = exp_list[n_train + n_val:] #测试部分的数据集
        #return  train_exps, val_exps

        return train_exps, val_exps, test_exps #返回训练值，监督值，测试值

    def find_max(self):#已掌握
        '''
        data scaler: max_abs_scaler
        val/test_targets: square #监督，测试目标：面积
        '''
        # caculate scales
        file_names = ['c_monitor.csv', 'fai.csv', 'T.csv',
                      'theta.csv', 'v.csv']#那几个文件的名称

        for file in file_names:
            abs_max = 0
            for exp in self.train_exps:
                path = os.path.join(self.data_folder, exp, file)
                df_exp = pd.read_csv(path, header=None)
                abs_max = max(np.max(df_exp.iloc[:,:-1].values), abs_max)

            self.scales.append(abs_max)

    def test_data_prepare(self):#未完全掌握
        for exp in self.test_exps:
            path = os.path.join(self.data_folder, exp)
            exp_inputs, exp_space, exp_time = \
                data_classify(path, self.input_seq_len,
                              self.pred_seq_len, tag='test')

            exp_inputs = np.stack(exp_inputs, axis=0)# 增加exp_inputs的维度
            exp_time = np.stack(exp_time, axis=0)# 增加exp_time的维度
            # data transform
            for i in range(len(self.scales)):
                exp_inputs[:, :, i, ...] /= self.scales[i]

            exp_time = pow(exp_time, 2)

            self.save_data(exp, exp_inputs, exp_time)

    def save_data(self, exp, exp_inputs, exp_time):#未完全掌握

        inputs_path = os.path.join(self.save_sample_path, 'test', exp, 'inputs')
        target_time_path = os.path.join(self.save_sample_path, 'test', exp,
                                        'target_time')

        create_dirs(inputs_path)
        create_dirs(target_time_path)

        for j in range(len(exp_inputs)):
            sample_name = 'sample_' + str(j) + '.pkl'

            with open(os.path.join(inputs_path, sample_name), 'wb') as f:
                pickle.dump(exp_inputs[j], f)

            with open(os.path.join(target_time_path, sample_name), 'wb') as f:
                pickle.dump(exp_time[j], f)


def data_transform(inputs, scales):#未完全掌握
    file_names = ['c_monitor.csv', 'fai.csv', 'T.csv',
                  'theta.csv', 'v.csv']
    for i in range(len(file_names)-1): # ？？？？？
        inputs[:, :, i, ...] /= scales[i]

    return inputs


def split(data_folder, exps, input_seq_len, pred_seq_len, save_sample_path,
          scales, tag):#未完全掌握
    '''
    concat every exp which is arranged as input, target_space,
    target_time respectively
    '''

    inputs_path = os.path.join(save_sample_path, tag, 'inputs')
    target_space_path = os.path.join(save_sample_path, tag, 'target_space')
    target_time_path = os.path.join(save_sample_path, tag, 'target_time')

    create_dirs(inputs_path)
    create_dirs(target_space_path)
    create_dirs(target_time_path)

    i = 0
    for exp in exps:
        path = os.path.join(data_folder, exp)
        inputs, target_space, target_time \
            = data_classify(path, input_seq_len, pred_seq_len, tag)
        inputs = data_transform(inputs, scales)


        for j in range(len(inputs)):
            sample_name = 'sample_'+str(j+i)+'.pkl'

            with open(os.path.join(inputs_path,sample_name), 'wb') as f:
                pickle.dump(inputs[j], f)

            with open(os.path.join(target_space_path, sample_name), 'wb') as f:
                pickle.dump(target_space[j], f)

            with open(os.path.join(target_time_path, sample_name), 'wb') as f:
                pickle.dump(target_time[j], f)

        i = i + len(inputs)


def data_classify(path, input_seq_len, pred_seq_len, tag):#未完全掌握
    '''
    To every exp, as input_seq_len=10, pred_seq_len=5 for an example:
    row_targets/row_inputs/test_inputs/test_space_targets = [
    [1,2,3...,10],
    [2,3,4,...11],
    ...
    [586,..,594,595]

    test_time=
    [
    [11,12,..15],
    [12,13,..16],
    ..
    [596,597,..600]
    ]

    train/val inputs space_targets = [
    [1,2,3...,10],
    [11,12,...20],
    ..

    ]
    train/val time_targets =[
    [11,12,..15],
    [21,22,..25],
    ...

    ]
    '''
    file_names = ['c_monitor.csv', 'c_target.csv', 'fai.csv', 'T.csv', 'theta.csv'
                  , 'v.csv']

    while '.DS_Store' in file_names:
        file_names.remove('.DS_Store')
    df_file = pd.read_csv(os.path.join(path, 'T.csv'), header=None)
    times = list(map(int, set(df_file.values[:, -1])))#提出除了最后一列(时间)之外的数据

    tar_file_name = 'c_target.csv'
    assert tar_file_name in file_names
    tar_df = pd.read_csv(os.path.join(path, tar_file_name), header=None)
    last_columns = tar_df[tar_df.columns[-1]]#提出目标浓度的最后一列，时间

    # targets
    row_targets = [np.float32(tar_df[last_columns == time].iloc[:, :-1].values)
                   for time in times]#提取除了最后一列之外的数据
    # row_targets = [tar_df[last_columns == time].values for time in times]

    # inputs
    file_names.remove('c_target.csv')
    dfs = [pd.read_csv(os.path.join(path, file), header=None)
           for file in file_names]
    row_inputs = []
    for time in times:
        cur_data = []
        for df in dfs:
            last_columns = df[df.columns[-1]]#提取最后一列的值，时间
            df = df[last_columns == time].iloc[:, :-1]#提出除了时间之外的数据
            # df = df[last_columns == time]
            cur_data.append(np.float32(df.values))
        row_inputs.append(np.stack(cur_data))

    inputs_ = [np.stack(row_inputs[i: i + input_seq_len])
               for i in range((len(times) - input_seq_len) + 1)]

    targets_ = [np.stack(row_targets[i: i + input_seq_len])
                for i in range((len(times) - input_seq_len) + 1)]

    if tag == 'test':
        for _ in range(pred_seq_len):
            inputs_.pop()
            targets_.pop()
        inputs = inputs_
        y_space = targets_

        for i in range(input_seq_len):
            row_targets.pop(0)
        y_time = [row_targets[i: i + pred_seq_len]
                  for i in range(len(row_targets) - pred_seq_len + 1)]

    elif tag == 'train' or tag == 'val':

        for _ in range(pred_seq_len):
            inputs_.pop()
            targets_.pop()

        # create blocks
        blocks_x = [inputs_[i::input_seq_len] for i in range(input_seq_len)]
        blocks_y = [targets_[i::input_seq_len] for i in range(input_seq_len)]

        # inputs
        inputs = sum(blocks_x, [])

        # space_targets
        y_space = sum(blocks_y, [])

        # time_targets
        for _ in range(input_seq_len):
            row_targets.pop(0)
        y_time_ = [np.stack(row_targets[i:i+pred_seq_len]) for i in
                range(len(row_targets)-pred_seq_len+1)]
        blocks_y_time = [y_time_[i::input_seq_len] for i in range(input_seq_len)]
        y_time = sum(blocks_y_time, [])
    else:
        raise Exception('no tag!')

    return np.stack(inputs), np.stack(y_space), np.stack(y_time)


def main():# 未完全掌握
    args = args_set('big')#解析输入参数，设立文件夹
    print('test')
    set_seeds(seed=1234, cuda=args.cuda)
    print(str(args.pred_seq_len/5)+'s')
    print(args.save_sample_path)#保存样本路径
    data_prepare = DataPrepare(data_folder=args.data_folder,
                               save_dir=args.save_dir,
                               save_sample_path=args.save_sample_path,
                               train_size=args.train_size,
                               val_size=args.val_size,
                               test_size=args.test_size,
                               input_seq_len=args.input_seq_len,
                               pred_seq_len=args.pred_seq_len, shuffle = True)#数据准备
    print(data_prepare.test_exps)
    print(data_prepare.train_exps)
    print(data_prepare.scales)
    print('train&val')
    data_prepare.create_data()
    print('test')
    data_prepare.test_data_prepare()




if __name__ == '__main__': #区分是自己作为自己的文件进行执行的，还是被导入到其他文件当作脚本使用
    main()