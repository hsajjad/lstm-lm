import utils
import argparse
import torch
import lstm_lm
import torch.optim as optim
import torch.nn as nn

def load_model(params, model, optimizer):
    #model.load_state_dict(torch.load(filepath))
    model.load_state_dict(params["model"])
    optimizer.load_state_dict(params["optimizer"])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM language model')

    parser.add_argument('--test', type=str, help='test data')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)

    args = parser.parse_args()

    # load dictionary
    word2idx = utils.read_json(args.vocab)
    # load data
    test = utils.process_valid_data(args.test, word2idx)
    # load model #
    params = torch.load(args.model)
    
    test_batches = utils.batchify(test, params["sequence_length"], params["batch_size"], word2idx)

    model = lstm_lm.LSTMLM(params)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.NLLLoss()
    load_model(params, model, optimizer)

    model.eval() # change state to evaluation mode
    print ("Test perplexity: ", utils.evaluate(model, loss_function, test_batches, params["use_gpu"])/len(test_batches))


  