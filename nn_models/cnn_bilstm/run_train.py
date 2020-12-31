import csv
import random
import numpy as np
import time
import torch
from model import Trainer, CNN_biLSTM


# training data filepath
training_data_filepath = '../../prepared_data/train_daily_30_step_15_data.csv'
validation_data_filepath = '../../prepared_data/dev_daily_30_step_15_data.csv'

save_model_path = './checkpoints/'


def read_data(filepath):
    inp_dataset = []
    tgt_dataset = []
    with open(filepath, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            inp_string = line[0]
            inp_list = [float(element) for element in inp_string.split()]

            tgt_string = line[1]
            tgt_scalar = float(tgt_string)

            inp_dataset.append(inp_list)
            tgt_dataset.append(tgt_scalar)

    return inp_dataset, tgt_dataset


def generate_batch(inp_dataset, tgt_dataset, batch_size=64):
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


def main(model, epochs, loss_fn, optimizer, lr_scheduler, train_inp_dataset, train_tgt_dataset, valid_inp_dataset, valid_tgt_dataset, generate_batch, prepare_input, device):
    # build Trainer
    trainer = Trainer(model, train_inp_dataset, train_tgt_dataset, valid_inp_dataset, valid_tgt_dataset, loss_fn, optimizer, lr_scheduler, generate_batch, prepare_input, device=device)

    # start training
    print('start training...\n')
    baseline_loss = 10000
    report = open('./report/train_info.csv', 'w')
    tsv_writer = csv.writer(report, delimiter='\t')
    for epoch in range(1, epochs+1):
        print('epoch {}'.format(epoch))
        start_time = time.time()
        train_loss, valid_loss = trainer.train(epoch=epoch)
        end_time = time.time()
        tsv_writer.writerow([epoch, train_loss, valid_loss])

        if valid_loss <= baseline_loss:
            trainer.save_model(path=save_model_path+'checkpoint{}.pt'.format(epoch+1))
            baseline_loss = valid_loss

        print('epoch {} cost {}s\n'.format(epoch, end_time-start_time))






if __name__ == '__main__':
    # read in training data
    print('reading dataset...')
    train_inp_dataset, train_tgt_dataset = read_data(training_data_filepath)
    valid_inp_dataset, valid_tgt_dataset = read_data(validation_data_filepath)

    # build model architecture
    print('building model...')
    device = 0
    model = CNN_biLSTM(device=device)
    model = model.to(device)

    # loss function & optimizer
    loss_fn = build_loss_function()
    optimizer = build_optimizer(model, learning_rate=0.002)
    lr_scheduler = build_lr_scheduler(optimizer, learning_rate_decay=0.95)

    # train model
    main(model, 30, loss_fn, optimizer, lr_scheduler, train_inp_dataset, train_tgt_dataset, valid_inp_dataset, valid_tgt_dataset, generate_batch, prepare_input, device)
