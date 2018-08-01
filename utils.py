import os
import json
import numpy as np
import codecs
import h5py
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def process_test_data_by_sentence(file_name, word2idx):
    data = []

    try:
        with codecs.open(file_name) as fp:
            for line in fp:
                data.append(line.strip().split() + ["eos"])
    except FileNotFoundError:
        print ("File does not exist: ", file_name)
        exit()

    print ("Size of data: ", len(data))
    test = []
    sent = []
    for sentence in data:
        sent = []
        for word in sentence:
            if word in word2idx:
                sent.append(word2idx[word])
            else:
                sent.append(word2idx["<unk>"])
        test.append(np.array(sent))

    return test

def process_valid_data(file_name, word2idx):
    
    data = []

    try:
        with codecs.open(file_name) as fp:
            for line in fp:
                data.append(line.strip().split() + ["eos"])
    except FileNotFoundError:
        print ("File does not exist: ", file_name)
        exit()

    print ("Size of data: ", len(data))
    test = []
    for sentence in data:
        for word in sentence:
            if word in word2idx:
                test.append(word2idx[word])
            else:
                test.append(word2idx["<unk>"])
    test = np.array(test, dtype=np.int)
    return test


def read_json(file_name):
    with open(file_name) as f:
        return json.load(f)

def write_json(data, file_name):
    f = open(file_name,"w")
    f.write(json.dumps(data))
    f.close()

def write_h5py(data, type, file_name):
    handle = h5py.File(file_name, 'w')
    handle.create_dataset(type, data=data)
    print ("Saved.. ", file_name)

def read_h5py(file_name, type):
    handle = h5py.File(file_name, 'r')
    return handle[type][:]


def batchify(data, max_sequence_length, batch_size, word2idx):    
    batch_sequence_length = max_sequence_length * batch_size

    if batch_sequence_length == len(data):
        batches = data.reshape(-1, batch_size, max_sequence_length)

    else:
        pad = np.array((batch_sequence_length - len(data)%(batch_sequence_length)) * [word2idx["eos"]], dtype=np.int)
        batches = np.concatenate((data, pad)).reshape((-1, batch_size, max_sequence_length))

    # batches is now [num_batches x sentences_in_minibatch x words_in_sentence]

    # torch expects words_in_sentence x sentences_in_minibatch
    # so, swap
    batches = np.swapaxes(batches, 1, 2)

    #print("Shape of data: ", batches.shape)
    return batches

def prepare_input(batch):
    tensor_ids = torch.from_numpy(batch)
    return tensor_ids


def evaluate(model, loss_function, batches, use_gpu):
    total_loss = 0
    for batch in tqdm(batches):
        model.zero_grad()
        model.hidden = model.init_hidden()

        X = prepare_input(batch[:-1,:])
        y = prepare_input(batch[1:,:])

        if use_gpu:
            X = X.cuda()
            y = y.cuda()
        
        output_scores = model(X)
        true_y = y.contiguous().view(-1, 1).squeeze()
        pred_y = output_scores.view(-1, model.vocab_size)

        loss = loss_function(pred_y, true_y)
        total_loss += loss.data

    return total_loss.cpu().numpy()

def predict(model, batch, idx2word, use_gpu): # batch of one word
    
    model.batch_size = 1
    model.zero_grad()
    model.hidden = model.init_hidden()

    tensor_ids = torch.from_numpy(batch[0,:])

    if use_gpu:
        tensor_ids = tensor_ids.cuda()

    output_scores = model(tensor_ids)

    # print (output_scores)
    pred_y = output_scores.view(-1, model.vocab_size)

    values, indices = torch.max(pred_y, 1)
    print(idx2word[indices[0].item()])

    return idx2word[indices[0].item()]