import utils
import argparse
import torch
import lstm_lm
import torch.optim as optim
import torch.nn as nn
import numpy as np

def load_model(params, model, optimizer):
    #model.load_state_dict(torch.load(filepath))
    model.load_state_dict(params["model"])
    optimizer.load_state_dict(params["optimizer"])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM language model')

    parser.add_argument('--test', type=str, help='test word to start generating from')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)

    args = parser.parse_args()

    # load dictionary
    word2idx = utils.read_json(args.vocab)
    idx2word = {value:key for key,value in word2idx.items()}

    # load model #
    params = torch.load(args.model)

# process input word
    test = [word2idx[args.test] if args.test in word2idx else word2idx["<unk>"]]
    test = np.array(test, dtype=np.int)
    print (test)
    test_batch =  utils.batchify(test, 1, 1, word2idx)
    print ("test batch", test_batch)
    model = lstm_lm.LSTMLM(params)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.NLLLoss()
    load_model(params, model, optimizer)

    model.eval() # change state to evaluation mode
    utils.predict(model, test_batch, idx2word)


  