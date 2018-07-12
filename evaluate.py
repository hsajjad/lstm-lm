import utils
import argparse
import torch

def load_model(filepath, model):
    model.load_state_dict(torch.load(filepath))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM language model')

    parser.add_argument('--test', type=str, help='test data')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load model
    model = load_model(args.model, model)
    model.eval() # change state to evaluation mode

    # load data
    word2idx = utils.read_json(args.vocab)
    test = utils.process_test_data(args.test, word2idx)
    
    # batchify
    for sentence in test:
        testbatch = utils.batchify(sentence, len(sentence), 1, word2idx)
        utils.evaluate(testbatch)


  