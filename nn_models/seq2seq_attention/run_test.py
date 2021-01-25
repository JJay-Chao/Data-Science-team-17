import csv
import os
import random
import numpy as np
import time
import torch
from model import Infer, Model


# training data filepath
# time series files
testing_data_series_filepath = '../data/series_data2/infer_64_64_step_32_series_data.csv'

# trend, seasonal, residual files
testing_data_trend_filepath = '../data/series_data2/infer_64_64_step_32_trend_data.csv'
testing_data_seasonal_filepath = '../data/series_data2/infer_64_64_step_32_seasonal_data.csv'
testing_data_residual_filepath = '../data/series_data2/infer_64_64_step_32_residual_data.csv'

# pageName file
pageName_filepath = '../data/series_data2/pageNames.csv'

# denormalize data
traffic_maxmin_filepath = '../data/series_data2/traffic_maxmin.csv'
pred_filepath = './pred2.csv'

# save model path
save_model_path = './checkpoints_second/checkpoint7.pt'


def read_maxmin(filepath):
    pageNames = []
    maxmin_data = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            pageName = line[0]
            max_value = line[1]
            min_value = line[2]

            pageNames.append(pageName)
            maxmin_data.append((float(max_value), float(min_value)))

    return pageNames, maxmin_data


def read_pageNames(filepath):
    pageNames = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            pageName = line[0]
            pageNames.append(pageName)

    return pageNames


def read_series_data(filepath, pageNames):
    series_data = {}
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line, pageName in zip(tsv_reader, pageNames):
            inp_series = [float(value) for value in line[0].split(' ')]

            if pageName not in series_data:
                series_data[pageName] = {'inp_series': [],
                                         'trend': [],
                                         'seasonal': [],
                                         'residual': []}

            series_data[pageName]['inp_series'].append(inp_series)

    return series_data


def read_trend_seasonal_residual_data(trend_path, seasonal_path, residual_path, series_data, pageNames):
    # trend
    with open(trend_path, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line, pageName in zip(tsv_reader, pageNames):
            trend_series = [float(value) for value in line[0].split(' ')]

            series_data[pageName]['trend'].append(trend_series)

    # seasonal
    with open(seasonal_path, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line, pageName in zip(tsv_reader, pageNames):
            seasonal_series = [float(value) for value in line[0].split(' ')]

            series_data[pageName]['seasonal'].append(seasonal_series)

    # residual
    with open(residual_path, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line, pageName in zip(tsv_reader, pageNames):
            residual_series = [float(value) for value in line[0].split(' ')]

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

    return data_array


def generate_batch(inp_dataset, batch_size=64):
    # generate batches
    pointer = 0
    input_batches = []
    while pointer+batch_size <= len(inp_dataset):
        input_batches.append(inp_dataset[pointer:pointer+batch_size])
        pointer += batch_size

    if pointer != len(inp_dataset):
        input_batches.append(inp_dataset[pointer:])

    return input_batches


def prepare_input(batches):
    # training data part
    test_batches = []
    for input_batch in batches:
        inpVar = torch.FloatTensor(input_batch)
        test_batches.append(inpVar)

    return test_batches


def main(model, batch_size, test_inp_array, generate_batch, prepare_input, device):
    # build Trainer
    inferer = Infer(model, batch_size, test_inp_array, generate_batch, prepare_input, device=device)

    # start training
    print('start inferring...\n')
    results = inferer.infer()

    return results


def write_results(pageNames, results, maxmin_data, filepath):
    with open(filepath, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for pageName, result, maxmin in zip(pageNames, results, maxmin_data):
            max_value = maxmin[0]
            min_value = maxmin[1]

            denormalize_values = []
            for value in result:
                value_std = (value - (-1)) / (1-(-1))
                value_org = (value_std * (max_value - min_value)) + min_value
                value_org = int(round(value_org))
                if value_org < 0:
                    value_org = 0

                denormalize_values.append(str(value_org))

            denormalize_string = ' '.join(denormalize_values)
            tsv_writer.writerow([pageName, denormalize_string])




if __name__ == '__main__':
    print('reading data...')
    pageNames, maxmin_data = read_maxmin(traffic_maxmin_filepath)

    if not os.path.isfile('../data/series_data2/test_inp_array.npy'):
        # training dictionary
        test_series_data = read_series_data(testing_data_series_filepath, pageNames)
        test_series_data = read_trend_seasonal_residual_data(testing_data_trend_filepath, testing_data_seasonal_filepath, testing_data_residual_filepath, test_series_data, pageNames)
        # test_series_data = read_country_agent_device_data(testing_data_country_filepath, testing_data_agent_filepath, testing_data_device_filepath, test_series_data)

        print('setting up data array...')
        # setup data array
        print('testing data:')
        test_inp_array = setup_data_array(test_series_data)

        # save to files
        np.save('../data/series_data2/test_inp_array.npy', test_inp_array)
    else:
        test_inp_array = np.load('../data/series_data2/test_inp_array.npy')

    print('testing data size: {}'.format(test_inp_array.shape))

    # build model framework
    print('building model architecture...\n')
    batch_size = 128
    device = torch.device("cuda:1")
    model = Model(batch_size=batch_size, device=device)
    model.load_state_dict(torch.load(save_model_path))
    model = model.to(device)
    print(model)

    # main function
    results = main(model, batch_size, test_inp_array, generate_batch, prepare_input, device)

    write_results(pageNames, results, maxmin_data, pred_filepath)
