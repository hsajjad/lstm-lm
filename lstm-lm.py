
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import codecs
import operator
import argparse
import h5py
import json
import os

def load_data(data_dir):
    try:
        train = read_h5py(data_dir+"/processed_train.h5", "train")
        valid = read_h5py(data_dir+"/processed_valid.h5", "valid")
        test = read_h5py(data_dir+"/processed_test.h5", "test")
        word2idx = read_json(data_dir+"/vocab.json")
        return train, valid, test, word2idx

    except FileNotFoundError:
        print ("processed_train.h5, valid, test or vocab.json do not exist in directory ", data_dir)
        exit()
    
    #idx2word = {value:key for key,value in word2idx.items()}
    return -1

def write_json(data, file_name):
    f = open(file_name,"w")
    f.write(json.dumps(data))
    f.close()

def read_json(file_name):
    with open(file_name) as f:
        return json.load(f)

def write_h5py(data, type, file_name):
    handle = h5py.File(file_name, 'w')
    handle.create_dataset(type, data=data)
    print ("Saved data in: "+file_name)

def read_h5py(file_name, type):
    handle = h5py.File(file_name, 'r')
    return handle[type][:]

# set input in the form of tensor
def prepare_input(batch):
     tensor_ids = torch.from_numpy(batch).to(device)
     return tensor_ids

def batchify(train, max_sequence_length, batch_size, word2idx):    
    batch_sequence_length = max_sequence_length*batch_size

    pad = np.array((batch_sequence_length - len(train)%(batch_sequence_length)) * [word2idx["eos"]], dtype=np.int)

    batches = np.concatenate((train, pad)).reshape((-1, batch_size, max_sequence_length))
    # batches is now [num_batches x sentences_in_minibatch x words_in_sentence]
    
    # torch expects words_in_sentence x sentences_in_minibatch
    # so, swap
    batches = np.swapaxes(batches, 1, 2)
    
    print("Shape of data: ", batches.shape)
    return batches

class LSTMLM(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, batch_size, layers, dropout):
        super(LSTMLM, self).__init__()
        
        self.hidden_size = hidden_size   
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        # added separate layers to have control over each layer
        self.lstms = nn.ModuleList([])
        self.lstms.append(nn.LSTM(embedding_size, hidden_size))
        for l in range(1,layers):
            #self.lstms.append(nn.dropout(dropout))
            self.lstms.append(nn.LSTM(hidden_size, hidden_size))

        self.hidden2output = nn.Linear(hidden_size, vocab_size)
        self.hidden = self.init_hidden()        

    def init_hidden(self):
        # set the dimenionaly of the hidden layer
        cell = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size)).to(device)
        hid = autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_size)).to(device)
        return (cell, hid)
    
    def forward(self, sequence):
        embeds = self.word_embeddings(sequence)
        lstm_outs = []
        hiddens = []
        lstm_out, self.hidden = self.lstms[0](embeds, self.hidden)
        lstm_outs.append(lstm_out)
        hiddens.append(self.hidden)
        for idx in range(1, len(self.lstms)):
            self.dropout(lstm_outs[idx-1])
            lstm_out, self.hidden = self.lstms[idx](lstm_outs[idx-1], hiddens[idx-1])
            lstm_outs.append(lstm_out)
            hiddens.append(self.hidden)
        output_space = self.hidden2output(lstm_outs[-1])
        output_scores = F.log_softmax(output_space, dim=2)
        return output_scores


def train_model(train_batches, word2idx, epochs, valid_batches, model_save):
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_batches, desc="Epoch %d/%d"%(epoch+1, epochs)):
            model.zero_grad()
            model.hidden = model.init_hidden()
            
            X = prepare_input(batch[:-1,:])
            y = prepare_input(batch[1:,:])

            output_scores = model(X).to(device)

            true_y = y.contiguous().view(-1, 1).squeeze()
            pred_y = output_scores.view(-1, len(word2idx))
            
            loss = loss_function(pred_y, true_y)
            total_loss += loss.data
            
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), model_save+"_"+str(epoch))

        print ("Training loss: ", total_loss.data.cpu().numpy()/len(train_batches))
        model.eval()
        print ("Validation loss: ", evaluate(valid_batches)/len(valid_batches))
        model.train()

#def load_model():
    ##Later to restore:
#model.load_state_dict(torch.load(filepath))
#model.eval()

def evaluate(batches):
    total_loss = 0
    for batch in tqdm(batches):
        model.zero_grad()
        model.hidden = model.init_hidden()

        X = prepare_input(batch[:-1,:])
        y = prepare_input(batch[1:,:])

        output_scores = model(X)
        true_y = y.contiguous().view(-1, 1).squeeze()
        pred_y = output_scores.view(-1, len(word2idx))

        loss = loss_function(pred_y, true_y)
        total_loss += loss.data

    return total_loss.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM language model')

    #parser.add_argument('--train', type=str, help='training data')
    #parser.add_argument('--validation', type=str, help='validation data')
    #parser.add_argument('--test', type=str, help='test data')
    #parser.add_argument('--vocab_size', type=int, default=10000)

    parser.add_argument('--data_dir', type=str, required=True, help='input path to the processed input files')
    parser.add_argument('--embedding_size', type=int, default=650, help='word embedding size')
    parser.add_argument('--rnn_size', type=int, default=650, help='hidden layer size')
    parser.add_argument('--sequence_length', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--output_model', type=str, required=True)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load data
    if not os.path.isdir(args.data_dir):
        print ("Error: input files directory does not exist")
        exit(0)

    train, valid, test, word2idx = load_data(args.data_dir)

    train_batches = batchify(train, args.sequence_length, args.batch_size, word2idx)
    valid_batches = batchify(valid, args.sequence_length, args.batch_size, word2idx)
    test_batches = batchify(test, args.sequence_length, args.batch_size, word2idx)
    
    # define loss, model and optimization
    model = LSTMLM(args.embedding_size, args.rnn_size, len(word2idx), args.batch_size, args.rnn_layers, args.dropout).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # model summary
    print (str(model))

    # training
    #nwords = len(train)
    train_model(train_batches, word2idx, args.epochs, valid_batches, args.output_model)

    # evaluate on test
    print ("Test loss: ", evaluate(test_batches)/len(test_batches))

