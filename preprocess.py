import os
import codecs
import argparse
import operator
import numpy as np
import utils

def process_train_data(file_name, vocab_size, output_dir):
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
    word2idx["<unk>"] = len(word2idx)
    idx2word[word2idx["<unk>"]] = "<unk>"
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
                train.append(word2idx["<unk>"])
    print (train[0:2])
    train = np.array(train, dtype=np.int)

    return train, word2idx, idx2word


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM language model')

    parser.add_argument('--train', type=str, help='training data', required=True)
    parser.add_argument('--validation', type=str, help='validation data')
    parser.add_argument('--test', type=str, help='test data')
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default=".")

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print ("Error: Output director does not exist")
        exit(0)

    # load data
    output_dir = args.output_dir

    train, word2idx, idx2word = process_train_data(args.train, args.vocab_size, output_dir)
    utils.write_h5py(train, "train", output_dir+"/processed_train.h5")
    utils.write_json(word2idx, output_dir+"/vocab.json")

    if args.validation != None:
        valid = utils.process_test_data(args.validation, word2idx)
        utils.write_h5py(valid, "valid", output_dir+"/processed_valid.h5")
    if args.test != None:
        test = utils.process_test_data(args.test, word2idx)
        utils.write_h5py(test, "test", output_dir+"/processed_test.h5")

