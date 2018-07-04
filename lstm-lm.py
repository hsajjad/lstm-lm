
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


## TODO 
# Check if parameters are same as saved for processed data

def process_train_data(file_path, vocab_size):
    
    word2idx = {}
    idx2word = {}
    train = []

    if os.path.exists("data/processed_train.h5") and os.path.exists("data/vocab.json"):
        print ("Loading processed files..")
        train = read_h5py("processed_train.h5")
        word2idx = read_json("vocab.json")
        idx2word = {value:key for key,value in word2idx.items()}

    else:
        print ("Processing files..")

        if file_path != "":    
            training_data = []
            train_vocab = {}
            with codecs.open(file_path) as fp:
                for line in fp:
                    words = line.strip().split()
                    for word in words:
                        if word not in train_vocab:
                            train_vocab[word] = 1
                        else:
                            train_vocab[word] +=1
                    training_data.append(words + ["eos"])
            
        else:    
            training_data = [
                ("The dog ate the apple eos".split()),
                ("Everybody read that book eos".split()),
                ("The cat ate the orange eos".split()),
                ("Everybody eat the apple eos".split())
            ]
            training_data = training_data * 50
            

        print ("Number of words in training: ", len(training_data))
        print ("A few training examples: ", training_data[0:3])

        # prune vocab based on vocab size
        sorted_train_vocab = sorted(train_vocab.items(), key=operator.itemgetter(1), reverse=True)
        pruned_vocab = sorted_train_vocab[0:vocab_size]
        pruned_vocab = ([i[0] for i in pruned_vocab])

        # create word2idx    
        word2idx["UNK"] = len(word2idx)
        idx2word[word2idx["UNK"]] = "UNK"
        word2idx["eos"] = len(word2idx)
        idx2word[word2idx["eos"]] = "eos"

        for idx, sent in enumerate(training_data):
            for word in sent:
                if word in pruned_vocab:
                    if word not in word2idx:
                        word2idx[word] = len(word2idx)
                        idx2word[word2idx[word]] = word
                    train.append(word2idx[word])
                else:
                    train.append(word2idx["UNK"])
                
        train = np.array(train, dtype=np.int)

        print(train.shape)
        if os.path.isdir("data"):
            write_h5py(train, "train", "processed_train.h5")
            write_json(word2idx, "vocab.json")

    return train, word2idx, idx2word

def write_json(data, file_name):
    f = open("data/"+file_name,"w")
    f.write(json.dumps(data))
    f.close()

def read_json(file_name):
    with open("data/"+file_name) as f:
        return json.load(f)

def write_h5py(data, data_name, file_name):
    handle = h5py.File("data/"+file_name, 'w')
    handle.create_dataset(data_name, data=data)
    print ("Saved data in: data/"+file_name)

def read_h5py(file_name):
    handle = h5py.File("data/"+file_name, 'r')
    return handle["train"][:]


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


def train_model(train_batches, word2idx, epochs, valid_batches):
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

        print ("Training loss: ", total_loss.data.cpu().numpy()/len(train_batches))
        print ("Validation loss: ", evaluate(valid_batches)/len(valid_batches))

def process_test_data(file_name, word2idx):

    if file_name != "":
        
        test_data = []
        with codecs.open(file_name) as fp:
            for line in fp:
                test_data.append(line.strip().split() + ["eos"])
    else:        
        test_data = [
            ("The dog ate the apple eos".split()),
            ("Everybody read that book eos".split()),
            ("The cat ate the orange eos".split()),
            ("Everybody eat the apple eos".split())
        ]

    print ("Size of test data: ", len(test_data))
    test = []
    for sentence in test_data:
        for word in sentence:
            if word in word2idx:
                test.append(word2idx[word])
            else:
                test.append(word2idx["UNK"])
    test = np.array(test, dtype=np.int)

    return test

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
    #print ("Loss: ", total_loss.cpu().numpy()/len(batches))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM language model')

    parser.add_argument('--train', type=str, help='training data')
    parser.add_argument('--validation', type=str, help='validation data')
    parser.add_argument('--test', type=str, help='test data')
    parser.add_argument('--embedding_size', type=int, default=650, help='word embedding size')
    parser.add_argument('--rnn_size', type=int, default=650, help='hidden layer size')
    parser.add_argument('--sequence_length', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load data
    train, word2idx, idx2word = process_train_data(args.train, args.vocab_size)
    train_batches = batchify(train, args.sequence_length, args.batch_size, word2idx)

    valid = process_test_data(args.validation, word2idx)
    valid_batches = batchify(valid, args.sequence_length, args.batch_size, word2idx)

    test = process_test_data(args.test, word2idx)
    test_batches = batchify(test, args.sequence_length, args.batch_size, word2idx)
    

    # define loss, model and optimization
    model = LSTMLM(args.embedding_size, args.rnn_size, len(word2idx), args.batch_size, args.rnn_layers, args.dropout).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # model summary
    print (str(model))

    # training
    nwords = len(train)
    train_model(train_batches, word2idx, args.epochs, valid_batches)

    # evaluate on test
    print ("Test loss: ", evaluate(test_batches)/len(test_batches))

