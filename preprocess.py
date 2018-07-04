import os
import codecs
import argparse
import operator
import numpy as np
import h5py
import json

def process_train_data(file_name, vocab_size):
    word2idx = {}
    idx2word = {}
    train = []

    training_data = []
    train_vocab = {}
    try:
        with codecs.open(file_name) as fp:
            for line in fp:
                words = line.strip().split()
                for word in words:
                    if word not in train_vocab:
                        train_vocab[word] = 1
                    else:
                        train_vocab[word] +=1
                training_data.append(words + ["eos"])
    except FileNotFoundError:
        print ("File does not exist: ", file_name)
        exit()
    
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

    write_h5py(train, "train", "processed_train.h5")
    write_json(word2idx, "vocab.json")

    return train, word2idx, idx2word

def process_test_data(file_name, word2idx, type=""):
    
    test_data = []

    try:
        with codecs.open(file_name) as fp:
            for line in fp:
                test_data.append(line.strip().split() + ["eos"])
    except FileNotFoundError:
        print ("File does not exist: ", file_name)
        exit()

    print ("Size of test data: ", len(test_data))
    test = []
    for sentence in test_data:
        for word in sentence:
            if word in word2idx:
                test.append(word2idx[word])
            else:
                test.append(word2idx["UNK"])
    test = np.array(test, dtype=np.int)

    write_h5py(test, type, "processed_"+type+".h5")


def write_json(data, file_name):
    f = open(file_name,"w")
    f.write(json.dumps(data))
    f.close()

def read_json(file_name):
    with open(file_name) as f:
        return json.load(f)

def write_h5py(data, data_name, file_name):
    handle = h5py.File(file_name, 'w')
    handle.create_dataset(data_name, data=data)
    print ("Saved.. ", file_name)

#def read_h5py(file_name):
 #   handle = h5py.File("data/"+file_name, 'r')
  #  return handle["train"][:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM language model')

    parser.add_argument('--train', type=str, help='training data', required=True)
    parser.add_argument('--validation', type=str, help='validation data')
    parser.add_argument('--test', type=str, help='test data')
    parser.add_argument('--vocab_size', type=int, default=10000)

    args = parser.parse_args()

    # load data
    train, word2idx, idx2word = process_train_data(args.train, args.vocab_size)
    if args.validation != None:
        process_test_data(args.validation, word2idx, type="valid")
    if args.test != None:
        process_test_data(args.test, word2idx, type="test")

