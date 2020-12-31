import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, device=0):
        super(CNN_LSTM, self).__init__()

        self.cnn_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Dropout(0.1))

        self.cnn_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.1))

        self.cnn_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1))

        self.lstm_1 = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, bias=True, bidirectional=False, batch_first=True)

        self.classifier = nn.Linear(in_features=32, out_features=1, bias=True)

        self.device = device


    def forward(self, batch):
        batch = batch.unsqueeze(1)

        # Convolutional Neural Network
        mapping_1 = self.cnn_layer1(batch)
        mapping_2 = self.cnn_layer2(mapping_1)
        mapping_3 = self.cnn_layer3(mapping_2)

        # LSTM
        mapping_3 = torch.transpose(mapping_3, 1, 2)
        _, state = self.lstm_1(mapping_3)
        hidden_state = state[0].squeeze(0)

        # Linear layer
        pred_flow = self.classifier(hidden_state)

        return pred_flow


class Trainer:
    def __init__(self, model, train_inp_dataset, train_tgt_dataset, valid_inp_dataset, valid_tgt_dataset, loss_fn, optimizer, lr_scheduler, generate_batch, prepare_input, device):
        self.model = model

        self.train_inp_dataset = train_inp_dataset
        self.train_tgt_dataset = train_tgt_dataset
        self.valid_inp_dataset = valid_inp_dataset
        self.valid_tgt_dataset = valid_tgt_dataset

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler

        self.generate_batch = generate_batch
        self.prepare_input = prepare_input

        self.device = device


    def train(self, epoch):
        # generate batch data
        train_inp_batches, train_tgt_batches = self.generate_batch(self.train_inp_dataset, self.train_tgt_dataset, batch_size=64)
        valid_inp_batches, valid_tgt_batches = self.generate_batch(self.valid_inp_dataset, self.valid_tgt_dataset, batch_size=64)
        train_batches, valid_batches = self.prepare_input([train_inp_batches, train_tgt_batches], [valid_inp_batches, valid_tgt_batches])

        train_iter = iter(train_batches)
        valid_iter = iter(valid_batches)

        # training
        epoch_train_loss = 0
        global_train_step = 0
        for step_batch, batch in enumerate(train_iter):
            # empty gradient descent info
            self.optimizer.zero_grad()

            # input-output pair
            inp_batch = batch[0]
            tgt_batch = batch[1]

            inp_batch = inp_batch.to(self.device)
            tgt_batch = tgt_batch.to(self.device)

            # feed into the model
            pred_flow = self.model.forward(inp_batch)

            # calculate loss
            batch_loss = self.loss_fn(pred_flow.squeeze(1), tgt_batch)
            epoch_train_loss += batch_loss.tolist()
            global_train_step += 1

            # backpropagation & gradient descent
            batch_loss.backward()
            self.optimizer.step()

        epoch_train_loss /= global_train_step
        print('epoch {} training loss: {}'.format(epoch, epoch_train_loss))


        # validation
        epoch_valid_loss = 0
        global_valid_step = 0
        for step_batch, batch in enumerate(valid_iter):
            # input-output pair
            inp_batch = batch[0]
            tgt_batch = batch[1]

            inp_batch = inp_batch.to(self.device)
            tgt_batch = tgt_batch.to(self.device)

            # feed into the model
            pred_flow = self.model.forward(inp_batch)

            # calculate loss
            batch_loss = self.loss_fn(pred_flow.squeeze(1), tgt_batch)
            epoch_valid_loss += batch_loss.tolist()
            global_valid_step += 1

        epoch_valid_loss /= global_valid_step
        print('epoch {} validation loss: {}'.format(epoch, epoch_valid_loss))

        return epoch_train_loss, epoch_valid_loss


    def infer(self):
        pass


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


class Infer:
    def __init__(self, model, test_inp_dataset, generate_batch, prepare_input, device):
        self.model = model

        self.test_inp_dataset = test_inp_dataset

        self.generate_batch = generate_batch
        self.prepare_input = prepare_input

        self.device = device


    def infer(self):
        # generate batch data
        test_inp_batches = self.generate_batch(self.test_inp_dataset, batch_size=64)
        test_batches = self.prepare_input(test_inp_batches)

        test_iter = iter(test_batches)

        # infer
        result = []
        for step_batch, batch in enumerate(test_iter):
            inp_batch = batch.to(self.device)

            # predict 64 days continuously
            date = 0
            batch_result = []
            while date < 64:
                # feed into the model
                pred_flow = self.model.forward(inp_batch)

                # store the predicted result
                batch_result.append(pred_flow)

                # refresh the input sequence
                batch_ = []
                inp_batch = torch.cat((inp_batch, pred_flow), 1)
                for element in inp_batch[:, 1:].tolist():
                    batch_.append(element)

                del inp_batch
                inp_batch = torch.FloatTensor(batch_)
                inp_batch = inp_batch.to(self.device)

                torch.cuda.empty_cache()

                date += 1

            del inp_batch

            # concatenate the batch result
            indicator = batch_result[0]
            for id in range(1, len(batch_result)):
                indicator = torch.cat((indicator, batch_result[id]), 1)

            result.append(indicator.tolist())

        return result
