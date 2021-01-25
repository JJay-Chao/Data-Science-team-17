import torch
import torch.nn as nn
from numpy import random
import numpy as np


random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(4, 32, num_layers=1, batch_first=True, bidirectional=True)

        # transform encoder_outputs into encoder_features
        self.W_h = nn.Linear(32*2, 32*2, bias=False)

    #seq_lens should be in descending order
    def forward(self, input_embed):
        encoder_outputs, hidden = self.lstm(input_embed)
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 32*2)    # B * t_k x hidden_dim*2
        encoder_feature = self.W_h(encoder_feature)    # B * t_k x hidden_dim*2

        return encoder_outputs, encoder_feature, hidden


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(32*2, 32)
        self.reduce_c = nn.Linear(32*2, 32)

        self.relu = nn.ReLU()

    def forward(self, hidden):
        h, c = hidden    # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, 32*2)
        hidden_reduced_h = self.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, 32*2)
        hidden_reduced_c = self.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        self.decode_proj = nn.Linear(32*2, 32*2)
        self.v = nn.Linear(32*2, 1, bias=False)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s_t_1, encoder_outputs, encoder_feature):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_1)    # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()    # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)    # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded    # B * t_k x 2*hidden_dim

        e = self.tanh(att_features)    # B * t_k x 2*hidden_dim
        scores = self.v(e)    # B * t_k x 1
        scores = scores.view(-1, t_k)    # B x t_k

        attn_dist = self.softmax(scores)    # B x t_k

        attn_dist = attn_dist.unsqueeze(1)    # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)    # B x 1 x n
        c_t = c_t.view(-1, 32*2)    # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        return c_t, attn_dist


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()

        # decoder
        self.lstm = nn.LSTM(1, 32, num_layers=1, batch_first=True, bidirectional=False)

        # p_vocab
        self.out_1 = nn.Linear((32*4), (32*2), bias=True)
        self.out_2 = nn.Linear((32*2), 1, bias=True)

    def forward(self, y_t_1_embed, s_t_1,
                encoder_outputs, encoder_feature):

        # run decoder | o _ o |
        lstm_out, s_t = self.lstm(y_t_1_embed.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, 32),
                             c_decoder.view(-1, 32)), 1)    # B x 2*hidden_dim


        # attention mechanism | ~ _ ~ |
        c_t, attn_dist = self.attention_network(s_t_hat, encoder_outputs, encoder_feature)


        # generation | ^ _ ^ |
        out_input = torch.cat((s_t_hat, c_t), 1)
        output = self.out_1(out_input)
        output = self.out_2(output)

        return output, s_t


class Model(nn.Module):
    def __init__(self, batch_size, device):
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.device = device

        encoder = Encoder()
        reduce_state = ReduceState()
        decoder = Decoder()

        self.encoder = encoder
        self.reduce_state = reduce_state
        self.decoder = decoder

    def forward(self, inp_batch):
        encoder_outputs, encoder_feature, hidden = self.encoder.forward(inp_batch)
        s_t_1 = self.reduce_state.forward(hidden)

        #start training with for loop
        pred = []
        y_t_1 = None
        for di in range(64):
            if di == 0:
                sos = -1.
                y_t_1 = torch.tensor([sos])
                y_t_1 = y_t_1.expand(inp_batch.size(0))
                y_t_1 = y_t_1.unsqueeze(1).to(self.device)
            else:
                y_t_1 = pred_value

            pred_value, s_t_1 = self.decoder.forward(y_t_1, s_t_1,
                                                     encoder_outputs, encoder_feature)

            pred.append(pred_value)

        # concatenate predict results
        pred = torch.cat(pred, dim=-1)

        return pred


class Trainer:
    def __init__(self, model, batch_size,
                 train_inp_array, train_tgt_array, valid_inp_array, valid_tgt_array, loss_fn, optimizer, lr_scheduler, generate_batch, prepare_input, device):
        self.model = model
        self.batch_size = batch_size

        self.train_inp_array = train_inp_array
        self.train_tgt_array = train_tgt_array
        self.valid_inp_array = valid_inp_array
        self.valid_tgt_array = valid_tgt_array

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler

        self.generate_batch = generate_batch
        self.prepare_input = prepare_input

        self.device = device

        # generate batch data
        print('generate batches...\n')

        train_inp_batches, train_tgt_batches = self.generate_batch(self.train_inp_array, self.train_tgt_array, batch_size=self.batch_size)
        valid_inp_batches, valid_tgt_batches = self.generate_batch(self.valid_inp_array, self.valid_tgt_array, batch_size=self.batch_size)

        self.train_batches, self.valid_batches = self.prepare_input([train_inp_batches, train_tgt_batches], [valid_inp_batches, valid_tgt_batches])

    def train(self, epoch):
        # training
        # set to train mode
        self.model.train()

        epoch_train_loss = 0
        global_train_step = 0

        self.train_iter = iter(self.train_batches)
        self.valid_iter = iter(self.valid_batches)

        for step_batch, batch in enumerate(self.train_iter):
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
            batch_loss = self.loss_fn(pred_flow, tgt_batch)
            epoch_train_loss += batch_loss.tolist()
            global_train_step += 1

            # backpropagation & gradient descent
            batch_loss.backward()
            self.optimizer.step()

            if (step_batch+1)%1000 == 0:
                print('batch: {}/{}\tloss: {}'.format(step_batch+1, len(self.train_batches), batch_loss.tolist()))

        epoch_train_loss /= global_train_step


        # validation
        # set to eval mode
        self.model.eval()

        epoch_valid_loss = 0
        global_valid_step = 0
        for step_batch, batch in enumerate(self.valid_iter):
            # input-output pair
            inp_batch = batch[0]
            tgt_batch = batch[1]

            inp_batch = inp_batch.to(self.device)
            tgt_batch = tgt_batch.to(self.device)

            # feed into the model
            pred_flow = self.model.forward(inp_batch)

            # calculate loss
            batch_loss = self.loss_fn(pred_flow, tgt_batch)
            epoch_valid_loss += batch_loss.tolist()
            global_valid_step += 1

        epoch_valid_loss /= global_valid_step

        return epoch_train_loss, epoch_valid_loss


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


class Infer:
    def __init__(self, model, batch_size,
                 test_inp_array, generate_batch, prepare_input, device):
        self.model = model
        self.batch_size = batch_size

        self.test_inp_array = test_inp_array

        self.generate_batch = generate_batch
        self.prepare_input = prepare_input

        self.device = device

        # generate batch data
        print('generate batches...\n')

        test_inp_batches = self.generate_batch(self.test_inp_array, batch_size=self.batch_size)

        self.test_batches = self.prepare_input(test_inp_batches)

        self.test_iter = iter(self.test_batches)


    def infer(self):
        # set to eval mode
        self.model.eval()

        results = []
        for step_batch, batch in enumerate(self.test_iter):
            # input-output pair
            input_batch = batch.to(self.device)

            # feed into the model
            pred_flow = self.model.forward(input_batch)

            results.append(pred_flow.tolist())

        results = np.concatenate(results, axis=0)

        return results
