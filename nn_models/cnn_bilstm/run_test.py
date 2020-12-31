import csv
import random
import numpy as np
import time
import torch
from model import Infer, CNN_biLSTM


# training data filepath
infer_data_filepath = '../../prepared_data/infer_daily_30_step_15_data.csv'
webpage_header_filepath = '../../prepared_data/infer_headers.csv'
webpage_maxmin_filepath = '../../prepared_data/web_traffic_maxmin.csv'

# selected model
save_model_path = "./checkpoints/checkpoint6.pt"

# write out the prediction result
save_predict_path = './report/pred.csv'


def read_data(filepath):
    inp_dataset = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            inp_string = line[0]
            inp_list = [float(element) for element in inp_string.split()]

            inp_dataset.append(inp_list)

    return inp_dataset


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


def prepare_input(test_batches):
    # testing data part
    test_input_batch = []
    for input_batch in test_batches:
        inpVar = torch.FloatTensor(input_batch)
        test_input_batch.append(inpVar)

    return test_input_batch


def main(model, test_inp_dataset, generate_batch, prepare_input, device):
    # build Tester
    inferer = Infer(model, test_inp_dataset, generate_batch, prepare_input, device=device)

    # start infer
    print('start infer...')
    result = inferer.infer()

    return result


def read_webpage_header_file(filepath):
    webpage_header = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for header in tsv_reader:
            webpage_header.append(header[0])

    return webpage_header


def read_webpage_maxmin_file(filepath):
    webpage_maxmin = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for pair in tsv_reader:
            webpage_maxmin.append(pair)

    return webpage_maxmin


def denormalize(result, webpage_maxmin):
    denormalized_result = []
    id = 0
    for batch in result:
        for instance in batch:
            temp = []
            for value in instance:
                max_flow = float(webpage_maxmin[id][0])
                min_flow = float(webpage_maxmin[id][1])

                denormalized_value = ((value * (max_flow-min_flow)) + min_flow)
                denormalized_value = round(denormalized_value)
                if denormalized_value < 0:
                    denormalized_value = 0
                temp.append(denormalized_value)

            denormalized_result.append(temp)

    return denormalized_result


def write_file(webpage_header, result, filepath):
    print('webpage_header has length: {}'.format(len(webpage_header)))
    print('result has length: {}'.format(len(result)))
    with open(filepath, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for header, instance in zip(webpage_header, result):
            instance_lst = [str(value) for value in instance]
            instance_string = ' '.join(instance_lst)
            tsv_writer.writerow([header, instance_string])





if __name__ == '__main__':
    # read in training data
    print('reading dataset...')
    test_inp_dataset = read_data(infer_data_filepath)

    # build model architecture
    print('building model...')
    device = 0
    model = CNN_biLSTM(device=device)
    model.load_state_dict(torch.load(save_model_path))
    model = model.to(device)

    # start infer
    result = main(model, test_inp_dataset, generate_batch, prepare_input, device=device)

    # write final result
    webpage_header = read_webpage_header_file(webpage_header_filepath)
    webpage_maxmin = read_webpage_maxmin_file(webpage_maxmin_filepath)
    denormalized_result = denormalize(result, webpage_maxmin)
    write_file(webpage_header, denormalized_result, save_predict_path)
