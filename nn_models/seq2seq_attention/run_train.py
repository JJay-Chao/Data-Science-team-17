import csv
import os
import random
import numpy as np
import time
import torch
from model import Trainer, Model

from tensorboardX import SummaryWriter


# training data filepath
# time series files
training_data_series_filepath = '../data/series_data2/train_64_64_step_32_series_data.csv'
validation_data_series_filepath = '../data/series_data2/valid_64_64_step_32_series_data.csv'

# trend, seasonal, residual files
training_data_trend_filepath = '../data/series_data2/train_64_64_step_32_trend_data.csv'
validation_data_trend_filepath = '../data/series_data2/valid_64_64_step_32_trend_data.csv'

training_data_seasonal_filepath = '../data/series_data2/train_64_64_step_32_seasonal_data.csv'
validation_data_seasonal_filepath = '../data/series_data2/valid_64_64_step_32_seasonal_data.csv'

training_data_residual_filepath = '../data/series_data2/train_64_64_step_32_residual_data.csv'
validation_data_residual_filepath = '../data/series_data2/valid_64_64_step_32_residual_data.csv'

# country, agent, device files
training_data_country_filepath = '../data/series_data2/train_country_data.csv'
training_data_agent_filepath = '../data/series_data2/train_agent_data.csv'
training_data_device_filepath = '../data/series_data2/train_device_data.csv'


# save model path
save_model_path = './checkpoints_second/'


def read_series_data(filepath):
    series_data = {}
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            pageName = line[0]
            inp_series = [float(value) for value in line[1].split(' ')]
            tgt_series = [float(value) for value in line[2].split(' ')]

            if pageName not in series_data:
                series_data[pageName] = {'inp_series': [],
                                         'tgt_series': [],
                                         'trend': [],
                                         'seasonal': [],
                                         'residual': []}

            series_data[pageName]['inp_series'].append(inp_series)
            series_data[pageName]['tgt_series'].append(tgt_series)

    return series_data


def read_trend_seasonal_residual_data(trend_path, seasonal_path, residual_path, series_data):
    # trend
    with open(trend_path, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            pageName = line[0]
            trend_series = [float(value) for value in line[1].split(' ')]

            series_data[pageName]['trend'].append(trend_series)

    # seasonal
    with open(seasonal_path, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            pageName = line[0]
            seasonal_series = [float(value) for value in line[1].split(' ')]

            series_data[pageName]['seasonal'].append(seasonal_series)

    # residual
    with open(residual_path, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            pageName = line[0]
            residual_series = [float(value) for value in line[1].split(' ')]

            series_data[pageName]['residual'].append(residual_series)

    return series_data


def read_country_agent_device_data(country_path, agent_path, device_path, series_data):
    # country
    with open(country_path, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            pageName = line[0]
            country_onehot = [int(value) for value in line[1].split(' ')]

            series_data[pageName]['country'] = country_onehot

    # agent
    with open(agent_path, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            pageName = line[0]
            agent_onehot = [int(value) for value in line[1].split(' ')]

            series_data[pageName]['agent'] = agent_onehot

    # device
    with open(device_path, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            pageName = line[0]
            device_onehot = [int(value) for value in line[1].split(' ')]

            series_data[pageName]['device'] = device_onehot

    return series_data


def setup_data_array(series_data):
    # extract arrays
    series_inp_array = []
    series_tgt_array = []
    trend_array = []
    seasonal_array = []
    residual_array = []
    # country_array = []
    # agent_array = []
    # device_array = []
    for pageName, features in series_data.items():
        for series in features['inp_series']:
            series_inp_array.append(series)

        for series in features['tgt_series']:
            series_tgt_array.append(series)

        for trend in features['trend']:
            trend_array.append(trend)

        for seasonal in features['seasonal']:
            seasonal_array.append(seasonal)

        for residual in features['residual']:
            residual_array.append(residual)

        # for i in range(len(features['inp_series'])):
        #     country_array.append(features['country'])
        #     agent_array.append(features['agent'])
        #     device_array.append(features['device'])


    series_inp_array = np.array(series_inp_array)
    series_tgt_array = np.array(series_tgt_array)
    trend_array = np.array(trend_array)
    seasonal_array = np.array(seasonal_array)
    residual_array = np.array(residual_array)
    # country_array = np.array(country_array)
    # agent_array = np.array(agent_array)
    # device_array = np.array(device_array)

    # concatenate to a single data array
    series_inp_array = np.expand_dims(series_inp_array, axis=-1)
    trend_array = np.expand_dims(trend_array, axis=-1)
    seasonal_array = np.expand_dims(seasonal_array, axis=-1)
    residual_array = np.expand_dims(residual_array, axis=-1)

    data_array = np.concatenate((series_inp_array, trend_array, seasonal_array, residual_array), axis=-1)

    # # tile country, agent, device array
    # country_array = np.expand_dims(country_array, axis=1)
    # country_array = np.tile(country_array, (1, data_array.shape[1], 1))
    #
    # agent_array = np.expand_dims(agent_array, axis=1)
    # agent_array = np.tile(agent_array, (1, data_array.shape[1], 1))
    #
    # device_array = np.expand_dims(device_array, axis=1)
    # device_array = np.tile(device_array, (1, data_array.shape[1], 1))
    #
    # # final concatenate
    # data_array = np.concatenate((data_array, country_array, agent_array, device_array), axis=-1)

    return data_array, series_tgt_array


def generate_batch(inp_dataset, tgt_dataset, batch_size=128):
    # shuffle train_set
    temp = list(zip(inp_dataset, tgt_dataset))
    random.shuffle(temp)
    inp_set, tgt_set = zip(*temp)

    # generate batches
    pointer = 0
    input_batches = []
    target_batches = []
    while pointer+batch_size <= len(inp_set):
        input_batches.append(inp_set[pointer:pointer+batch_size])
        target_batches.append(tgt_set[pointer:pointer+batch_size])
        pointer += batch_size

    if pointer != len(inp_set):
        input_batches.append(inp_set[pointer:])
        target_batches.append(tgt_set[pointer:])

    return input_batches, target_batches


def prepare_input(train_batches, valid_batches):
    # training data part
    input_train_batches = train_batches[0]
    target_train_batches = train_batches[1]
    train_batches = []
    for input_batch, target_batch in zip(input_train_batches, target_train_batches):
        inpVar = torch.FloatTensor(input_batch)
        tgtVar = torch.FloatTensor(target_batch)
        train_batches.append((inpVar, tgtVar))

    # validation data part
    input_valid_batches = valid_batches[0]
    target_valid_batches = valid_batches[1]
    valid_batches = []
    for input_batch, target_batch in zip(input_valid_batches, target_valid_batches):
        inpVar = torch.FloatTensor(input_batch)
        tgtVar = torch.FloatTensor(target_batch)
        valid_batches.append((inpVar, tgtVar))

    return train_batches, valid_batches


def build_loss_function():
    loss_fn = torch.nn.L1Loss(reduction='mean')
    return loss_fn


def build_optimizer(model, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer


def build_lr_scheduler(optimizer, learning_rate_decay=0.95):
    lambda2 = lambda epoch: learning_rate_decay ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=[lambda2])
    return scheduler


def main(model, batch_size, epochs, loss_fn, optimizer, lr_scheduler, train_inp_array, train_tgt_array, valid_inp_array, valid_tgt_array, generate_batch, prepare_input, device):
    # build Trainer
    trainer = Trainer(model, batch_size, train_inp_array, train_tgt_array, valid_inp_array, valid_tgt_array, loss_fn, optimizer, lr_scheduler, generate_batch, prepare_input, device=device)

    # start training
    print('start training...\n')
    baseline_loss = 10000
    for epoch in range(epochs):
        print('epoch: {}'.format(epoch+1))

        start_time = time.time()
        train_loss, valid_loss = trainer.train(epoch=epoch)
        end_time = time.time()

        if valid_loss <= baseline_loss:
            trainer.save_model(path=save_model_path+'checkpoint{}.pt'.format(epoch+1))
            baseline_loss = valid_loss

        print('train_loss: {}   valid_loss: {}   cost {}s\n'.format(train_loss, valid_loss, end_time-start_time))

        # update tensorboard
        writer.add_scalar('Training/L1 loss', train_loss, epoch+1)
        writer.add_scalar('Validation/L1 loss', valid_loss, epoch+1)



if __name__ == '__main__':
    # open tensorboard online
    writer = SummaryWriter('tensorboard/64days_32stride_second')

    print('reading data...')
    if not os.path.isfile('../data/series_data2/train_inp_array.npy'):
        # training dictionary
        train_series_data = read_series_data(training_data_series_filepath)
        train_series_data = read_trend_seasonal_residual_data(training_data_trend_filepath, training_data_seasonal_filepath, training_data_residual_filepath, train_series_data)
        # train_series_data = read_country_agent_device_data(training_data_country_filepath, training_data_agent_filepath, training_data_device_filepath, train_series_data)

        # validation dictionary
        valid_series_data = read_series_data(validation_data_series_filepath)
        valid_series_data = read_trend_seasonal_residual_data(validation_data_trend_filepath, validation_data_seasonal_filepath, validation_data_residual_filepath, valid_series_data)
        # valid_series_data = read_country_agent_device_data(training_data_country_filepath, training_data_agent_filepath, training_data_device_filepath, valid_series_data)

        print('setting up data array...')
        # setup data array
        print('training data:')
        train_inp_array, train_tgt_array = setup_data_array(train_series_data)
        print('validation data:')
        valid_inp_array, valid_tgt_array = setup_data_array(valid_series_data)

        # save to files
        np.save('../data/series_data2/train_inp_array.npy', train_inp_array)
        np.save('../data/series_data2/train_tgt_array.npy', train_tgt_array)
        np.save('../data/series_data2/valid_inp_array.npy', valid_inp_array)
        np.save('../data/series_data2/valid_tgt_array.npy', valid_tgt_array)
    else:
        train_inp_array = np.load('../data/series_data2/train_inp_array.npy')
        train_tgt_array = np.load('../data/series_data2/train_tgt_array.npy')
        valid_inp_array = np.load('../data/series_data2/valid_inp_array.npy')
        valid_tgt_array = np.load('../data/series_data2/valid_tgt_array.npy')

    print('training data size: {}'.format(train_inp_array.shape))
    print('validation data size: {}\n'.format(valid_inp_array.shape))

    # build model framework
    print('building model architecture...\n')
    batch_size = 128
    device = torch.device("cuda:1")
    model = Model(batch_size=batch_size, device=device)
    model = model.to(device)
    print(model)

    # loss function & optimizer
    print('defining functions...\n')
    loss_fn = build_loss_function()
    optimizer = build_optimizer(model, learning_rate=0.002)
    lr_scheduler = build_lr_scheduler(optimizer, learning_rate_decay=0.001)

    # main function
    main(model, batch_size, 30, loss_fn, optimizer, lr_scheduler, train_inp_array, train_tgt_array, valid_inp_array, valid_tgt_array, generate_batch, prepare_input, device)
