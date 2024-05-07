import time
import os
import pickle
import random as r

import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pylab import *

from init_project import create_dirs, set_seeds, args_set
from spatialmodel import SpatialModel
from seq2seq_new import Seq2seq_new

from dataprepare_new import DataPrepare, data_classify
from evaluation import correlation_coefficient, compute_RMSE, accuracy, \
    fit_performance
#from sparce_test import data_trans

class Tester(object):
    def __init__(self, test_exps, data_folder, scales, input_seq_len,
                 pred_seq_len, model_spatial, model_time, extract_num, save_dir,
                 save_sample_path, device):

        self.test_exps = test_exps
        self.data_folder = data_folder
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.scales = scales
        self.model_spatial = model_spatial.to(device)
        self.model_time = model_time.to(device)
        self.device = device
        self.extract_num = extract_num
        self.save_dir = save_dir
        self.save_sample_path = save_sample_path

        self.y_pred_list = []
        self.y_target_list = []

        self.test_exps_num = 4
        self.doc = open(os.path.join(save_dir, 'out.txt'),'w')
        self.sparce = False
        self.MR = 0.6
        print('sparce:', self.sparce)


    def run_test_loop(self):

        self.model_time.batch_size = 1
        self.model_spatial.eval()
        self.model_time.eval()

        rmse = 0
        error = 0
        r_delta = 0
        accuracy_rate = 0

        t = []

        for exp in self.test_exps:
            input_path = os.path.join(self.save_sample_path, 'test',
                                      exp, 'inputs')
            target_time_path = os.path.join(self.save_sample_path, 'test',
                                            exp, 'target_time')

            length_data = len(os.listdir(input_path))
            y_pred_exp = []
            y_tar_exp = []

            exp_time = 0
            start_time = time.time()
            for i in range(length_data):
                input_sample_path = os.path.join(input_path,
                                                 'sample_{0}.pkl'.format(i))
                target_time_sample_path = os.path.join(target_time_path,
                                                 'sample_{0}.pkl'.format(i))
                with open(input_sample_path, 'rb') as f:
                    input_sample = pickle.load(f)

                with open(target_time_sample_path, 'rb') as f:
                    target_time_sample = pickle.load(f)

                # 是否稀疏化
                if self.sparce :
                    new_input_sample = []
                    for i in range(len(input_sample)):
                        new_input_sample.append(data_trans(input_sample[i],
                                                           self.MR))

                    input_i = torch.FloatTensor(new_input_sample)
                else:
                    input_i = torch.FloatTensor(input_sample)

                y_time_i = torch.FloatTensor(target_time_sample).squeeze(-1)

                y_pred_i, y_target_i = \
                    pred(self.model_spatial, self.model_time, input_i,
                              y_time_i, self.device)

                epoch_time = time.time()-start_time
                exp_time += epoch_time

                y_pred_exp.append(y_pred_i)
                y_tar_exp.append(y_target_i)
                start_time = time.time()

            t.append(exp_time/length_data)
            print(exp_time/length_data)

            rmse_exp, error_exp, r_exp, accuracy_rate_exp = \
                evaluation_index(y_pred_exp, y_tar_exp)

            rmse += rmse_exp
            error += error_exp
            r_delta += abs(1 - r_exp)
            accuracy_rate += accuracy_rate_exp

            self.y_pred_list.append(y_pred_exp)
            self.y_target_list.append(y_tar_exp)

        print(round(np.mean(t),6))
        # self.evaluation_mse_figure()
        print("Test RMSE: {0:.3f}".format(rmse / len(self.test_exps)))
        print("Test error: {0:.3f}".format(error / len(self.test_exps)))
        print("Test r: {0:.3f}".format(1 - (r_delta / len(self.test_exps))))
        print(
            "Test accuracy: {0:.3f}".format(accuracy_rate / len(self.test_exps))
            )
        print("Test RMSE: {0:.3f}".format(rmse/ len(self.test_exps)),
              file=self.doc)
        print("Test r: {0:.3f}".format(1 - (r_delta/ len(self.test_exps))), file=self.doc)
        print("Test accuracy: {0:.3f}".format(accuracy_rate/ len(self.test_exps))
              ,file=self.doc)
        # self.evaluation_fit_figure()
        # self.space_concentration_figure()
        #
        torch.save(self.y_pred_list, os.path.join(self.save_dir, 'y_pred_list'))
        torch.save(self.y_target_list, os.path.join(self.save_dir, 'y_tar_list'))


    def y_change(self, y_exp, random_extraction_seq):
        '''
        map data in real space
        '''
        y_out = []
        for i in range(self.pred_seq_len):
            y_time = y_exp[random_extraction_seq, i, :]
            y_insert_time = np.insert(y_time, 263, 0)
            y_insert_time = np.insert(y_insert_time, 527, 0)
            y_insert_time = np.insert(y_insert_time, 791, 0)
            y_change = y_insert_time.reshape(24, 11, -1)
            y_out.append(y_change)
        return y_out

def pred(model_spatial, model_time, input_data, y_time, device):

    y_pred_spatial = model_spatial(input_data.to(device))
    # y_pred = model_time(y_pred_spatial, z_tar=y_time)
    # epoch_start = time.time()
    y_pred = model_time(y_pred_spatial, device=device)
    y_pred = pow(y_pred, 2)

    y_time = y_time.cpu()
    y_pred = y_pred.squeeze().cpu()
    # epoch_end = time.time()
    y_pre = Variable(y_pred).squeeze().numpy()
    y_tar = Variable(y_time).squeeze().numpy()

    # print(epoch_end-epoch_start, 's')

    return y_pre, y_tar

def evaluation_index(y_pred, y_tar):
    y_pred_cat = y_pred[0].reshape(-1)
    y_target_cat = y_tar[0].reshape(-1)
    for j in range(1, len(y_pred)):
        y_pred_cat = np.concatenate((y_pred_cat,
                                         y_pred[j].reshape(-1)))
        y_target_cat = np.concatenate((y_target_cat,
                                           y_tar[j].reshape(-1)))

    rmse,error = compute_RMSE(y_pred_cat, y_target_cat)
    r = correlation_coefficient(y_pred_cat, y_target_cat)
        # fb = FB(y_pred_cat, y_target_cat)
    accuracy_rate = accuracy(y_pred_cat, y_target_cat)

    return rmse, error, r, accuracy_rate

# def evaluation_index(y_pred, y_tar):
#     r = 0
#     rmse = 0
#     error = 0
#     acc= 0
#     l=len(y_pred)
#     for j in range(l):
#
#         rmse_s,error_s = compute_RMSE(y_pred[j].reshape(-1),  y_tar[j].reshape(-1))
#         rmse += rmse_s
#         error += error_s
#         r_s = correlation_coefficient(y_pred[j].reshape(-1),  y_tar[j].reshape(-1))
#         r += r_s
#         # fb = FB(y_pred_cat, y_target_cat)
#         acc_s = accuracy(y_pred[j].reshape(-1),  y_tar[j].reshape(-1))
#         acc += acc_s
#
#     return rmse/l, error/l, r/l, acc/l


def main():

    args = args_set('big')
    print(args.pred_seq_len)
    print(args.save_dir)

    # Create save dir
    create_dirs(args.save_dir)

    # Check CUDA
    if torch.cuda.is_available():
        args.cuda = True
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))

    # Set seeds
    set_seeds(seed=1234, cuda=args.cuda)

    # load state
    model_spatial = SpatialModel(num_input_channels=5,
                                 out_num=1053,
                                 dropout_p=args.dropout_p)

    # model_time = Seq2seq(num_features=1053,
    #                      hidden_size=512, input_seq_len=args.input_seq_len,
    #                      pred_seq_len=args.pred_seq_len,
    #                      batch_size=1)
    model_time = Seq2seq_new(num_features=1053,
                             hidden_size=512,
                             input_seq_len=args.input_seq_len,
                             pred_seq_len=args.pred_seq_len,
                             batch_size=1)
    # model_time = Seq2seq_attn(num_features=1053,
    #                           input_seq_len=args.input_seq_len,
    #                           pred_seq_len=args.pred_seq_len,
    #                           batch_size=1,
    #                           dropout=args.dropout_p)
    # model_time = Seq2seq_mlp(num_features=1053,
    #                          input_seq_len=args.input_seq_len,

    #                          pred_seq_len=args.pred_seq_len,
    #                          batch_size=1, device=args.device)

    resume = os.path.join(args.save_dir, 'check_point_{}'.format(50))
    print('Resuming model check point from {}\n'.format(50))
    check_point = torch.load(resume)
    model_spatial.load_state_dict(check_point['model_spatial'])
    model_spatial.to(args.device)
    model_time.load_state_dict(check_point['model_time'])
    model_time.to(args.device)

    # data = DataPrepare(save_dir=args.save_dir, data_folder=args.data_folder,
    #                    train_size=args.train_size,
    #                    val_size=args.val_size,
    #                    test_size=args.test_size,
    #                    input_seq_len=args.input_seq_len,
    #                    pred_seq_len=args.pred_seq_len, shuffle=True)
    # data.create_data()

    test_exps = np.load('test_exp_list.npy')
    scales = np.load('scales.npy')

    tester = Tester(test_exps=test_exps, data_folder=args.data_folder,
                    scales=scales, input_seq_len=args.input_seq_len,
                    pred_seq_len=args.pred_seq_len,
                    model_spatial=model_spatial,
                    model_time=model_time, extract_num=4,
                    save_dir = args.save_dir,
                    save_sample_path=args.save_sample_path,
                    device='cuda')
    print('sparce:',tester.sparce)
    print('MR:', tester.MR)
    tester.run_test_loop()




if __name__ == '__main__':
    main()