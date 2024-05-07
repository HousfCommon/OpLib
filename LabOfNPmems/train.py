import copy
import random
import time

import torch
import torch.nn as nn
import numpy as np


def collate_fn(batch):
    '''
    :param batch: type:list, len(batch)=batch_size, the elements's type in batch is Series
    :return: the processed batch
    '''
    # Make a deep copy
    batch_copy = copy.deepcopy(batch)
    processed_batch = {"input": [], "target_space": [], 'target_time': []}

    input_data = []
    target_space = []
    target_time = []

    l = len(batch_copy)
    random_list = [i for i in range(l)]
    np.random.shuffle(random_list)

    for i in random_list:
        input_seq = batch_copy[i][0]
        target_space_seq = batch_copy[i][1]
        target_time_seq = batch_copy[i][2]
        for j in range(len(input_seq)):
            input_data.append(input_seq[j])
            target_space.append(target_space_seq[j])
        target_time.append(target_time_seq)

    processed_batch['input'] = torch.FloatTensor(input_data)
    processed_batch['target_space'] = torch.FloatTensor(target_space).squeeze(-1)
    processed_batch['target_time'] = torch.FloatTensor(target_time).squeeze(-1)

    return processed_batch

# class My_loss(nn.Module):
#     def __init__(self, beta):
#         super().__init__()
#         self.beta = beta
#
#     def forward(self, y_pred, y_tar):
#         part1 = torch.mean(torch.pow((y_pred - y_tar), 2))
#         part2 = torch.mean(torch.pow((y_pred - y_tar)/ y_tar, 2))
#         return part1 + self.beta * part2


class Trainer(object):
    def __init__(self, dataset, model_spatial, model_time, optimizer, scheduler,
                 device, teacher_forcing_ratio, train_state):
        self.dataset = dataset
        self.model_spatial = model_spatial.to(device)
        self.model_time = model_time.to(device)
        self.device = device
        # self.loss_func = nn.SmoothL1Loss()

        self.loss_func = nn.MSELoss()
        # self.loss_func = My_loss(beta=0.5)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.train_state = train_state


    def update_train_state(self):

        print(
        "[EPOCH]: {0} | [LR]: {1:.7f} | [TRAIN LOSS]: {2:.7f} | [VAL LOSS]: {3:.3f}"
        .format(
            self.train_state['epoch_index'], self.train_state['learning_rate'],
            self.train_state['train_loss'][-1],
            self.train_state['val_loss'][-1]))

        # Save one model at least
        if self.train_state['epoch_index'] == 1:
            self.train_state['stop_early'] = False

        elif self.train_state['epoch_index'] > 1:
            loss_t = self.train_state['val_loss'][-1]

            # If loss worsened
            if loss_t >= self.train_state['early_stopping_best_val']:
                # Update step
                self.train_state['early_stopping_step'] += 1

            # Loss decreased
            else:
                self.train_state['early_stopping_step'] = 0

            # Stop early
            self.train_state['stop_early'] = \
                self.train_state['early_stopping_step']\
                >= self.train_state['early_stopping_criteria']

        return self.train_state


    def run_train_loop(self, batch_generator, alpha, device):

        running_loss = 0
        self.model_spatial.train()
        self.model_time.train()
        # start_time = time.time()
        for batch_index, batch_dict in enumerate(batch_generator):

            # if (batch_index+1) % 20 == 0:
            #     print(str(batch_index+1)+ ':'+str((time.time()-start_time)/60)+'min')
            #     start_time = time.time()
            # print('It costs {:.2f} mili-seconds for data loading'.format(
            #     1000 * (time.time() - start_time)))

            # zero the gradients
            self.optimizer.zero_grad()

            # cur_start = time.time()

            # compute the output
            y_pred_space = self.model_spatial(batch_dict['input'])
            # space_end = time.time()
            # print('It costs {:.2f} mili-seconds for spatial prediction'.format(
            #     1000 * (space_end - cur_start)))

            y_pred = self.model_time(y_pred_space, device=device)
            # time_end = time.time()
            # print('It costs {:.2f} mili-seconds for time prediction'.format(
            #     1000 * (time_end - space_end)))

            loss_space = self.loss_func(y_pred_space,
                                        batch_dict['target_space'])
            # space_loss_end = time.time()
            # print('It costs {:.2f} mili-seconds for space loss'.format(
            #     1000 * (space_loss_end - time_end)))

            loss_time = self.loss_func(y_pred, batch_dict['target_time'])
            # time_loss_end = time.time()
            # print('It costs {:.2f} mili-seconds for time loss'.format(
            #     1000 * (time_loss_end - space_loss_end)))

            loss = alpha * loss_space + loss_time

            # loss_t = loss_time.item()
            # running_loss += loss_t * batch_dict['target_time'].size(0)
            # loss.backward()
            # back_end = time.time()
            # print('It costs {:.2f} mili-seconds for backward'.format(
            #     1000 * (back_end - time_loss_end)))

            # loss = self.loss_func(y_pred, batch_dict['target_time'])
            #
            #
            loss_t = loss.item()
            running_loss += loss_t * batch_dict['target_time'].size(0)
            loss.backward()

            self.optimizer.step()
            # optim_step_end = time.time()
            # print('It costs {:.2f} mili-seconds for time prediction'.format(
            #     1000 * (optim_step_end - back_end)))

            # start_time = time.time()


        self.train_state['train_loss'].append(
            running_loss/ self.dataset.sample_num)

    def run_val_loop(self, batch_generator, device):

        running_loss = 0.

        self.model_spatial.eval()
        self.model_time.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output

            y_pred_space = self.model_spatial(batch_dict['input'])
            y_pred = self.model_time(y_pred_space, device=device)
            # y_pred = self.model_time(y_pred_space, z_tar=[],
            #                          device=device, use_teacher_forcing=False)

            loss = self.loss_func(y_pred, batch_dict['target_time'])
            loss_t = loss.item()
            running_loss += loss_t * batch_dict['target_time'].size(0)

        self.train_state['val_loss'].append(
            running_loss/ self.dataset.sample_num)
        self.train_state['learning_rate'] = self.optimizer.param_groups[1]['lr']

        self.train_state = self.update_train_state()
        # self.scheduler.step()
        self.scheduler.step(self.train_state['val_loss'][-1])

    #
    #
    # def save_train_state(self):
    #     self.train_state["done_training"] = True
    #     with open(os.path.join(self.save_dir, "train_state.json"), "w") as fp:
    #         json.dump(self.train_state, fp)