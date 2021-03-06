
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
import utils

def load_data(data_dir):
    try:
        train = utils.read_h5py(data_dir+"/processed_train.h5", "train")
        valid = utils.read_h5py(data_dir+"/processed_valid.h5", "valid")
        test = utils.read_h5py(data_dir+"/processed_test.h5", "test")
        word2idx = utils.read_json(data_dir+"/vocab.json")
        return train, valid, test, word2idx

    except FileNotFoundError:
        print ("processed_train.h5, valid, test or vocab.json do not exist in directory ", data_dir)
        exit()
    
    #idx2word = {value:key for key,value in word2idx.items()}
    return -1

class LSTMLM(nn.Module):
    def __init__(self, params): #embedding_size, hidden_size, vocab_size, batch_size, layers, dropout, use_gpu):
        super(LSTMLM, self).__init__()
        
        self.use_gpu = params["use_gpu"]
        self.hidden_size = params["rnn_size"]
        self.vocab_size = params["vocab_size"] 
        self.batch_size = params["batch_size"]
        self.embedding_size = params["embedding_size"]
        self.dropout = params["dropout"]
        self.num_layers = params["rnn_layers"]
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout)

        self.hidden2output = nn.Linear(self.hidden_size, self.vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # set the dimenionaly of the hidden layer
        cell = autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        hid = autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        if self.use_gpu:
            cell = cell.cuda()
            hid = hid.cuda()
        return (cell, hid)
    
    def forward(self, sequence):
        embeds = self.word_embeddings(sequence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        output_space = self.hidden2output(lstm_out)
        output_scores = F.log_softmax(output_space, dim=2)
        return output_scores

def train_model(train_batches, word2idx, epochs, valid_batches, model_save, params, use_gpu):
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_batches, desc="Epoch %d/%d"%(epoch+1, epochs)):
            model.zero_grad()
            model.hidden = model.init_hidden()
            
            X = utils.prepare_input(batch[:-1,:])
            y = utils.prepare_input(batch[1:,:])

            if use_gpu:
                X = X.cuda()
                y = y.cuda()

            output_scores = model(X)

            true_y = y.contiguous().view(-1, 1).squeeze()
            pred_y = output_scores.view(-1, len(word2idx))
            
            loss = loss_function(pred_y, true_y)
            total_loss += loss.data
            
            loss.backward()
            optimizer.step()

        params["model"] = model.state_dict()
        params["optimizer"] = optimizer.state_dict()
        params["epoch"] = epoch
        torch.save(params, model_save+"_"+str(epoch))

        print ("Training loss: ", total_loss.data.cpu().numpy()/len(train_batches))
        model.eval()
        print ("Validation loss: ", utils.evaluate(model, loss_function, valid_batches, use_gpu)/len(valid_batches))
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM language model')

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

    use_gpu = torch.cuda.is_available()

    # load data
    if not os.path.isdir(args.data_dir):
        print ("Error: input files directory does not exist")
        exit(0)

    params = {}
    params["embedding_size"] = args.embedding_size
    params["rnn_size"] = args.rnn_size
    params["rnn_layers"] = args.rnn_layers
    params["dropout"] = args.dropout
    params["use_gpu"] = use_gpu
    params["sequence_length"] = args.sequence_length
    params["batch_size"] = args.batch_size

    train, valid, test, word2idx = load_data(args.data_dir)
    params["vocab_size"] = len(word2idx)

    train_batches = utils.batchify(train, args.sequence_length, args.batch_size, word2idx)
    valid_batches = utils.batchify(valid, args.sequence_length, args.batch_size, word2idx)
    test_batches =  utils.batchify(test, args.sequence_length, args.batch_size, word2idx)
    
    # define loss, model and optimization
    model = LSTMLM(params)
    if use_gpu:
        print ("CUDA found!")
        model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # model summary
    print (model)

    # training
    train_model(train_batches, word2idx, args.epochs, valid_batches, args.output_model, params, use_gpu)

    # evaluate on test
    print ("Test loss: ", utils.evaluate(model, loss_function, test_batches, use_gpu)/len(test_batches))

