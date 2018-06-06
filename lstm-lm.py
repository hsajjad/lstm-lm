
# coding: utf-8

# In[118]:

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm_notebook

import codecs


# In[173]:

EMBEDDING_SIZE = 100
HIDDEN_SIZE = 50
MAX_SEQ_LENGTH = 30
BATCH_SIZE = 32
VOCAB_SIZE = 10000


# In[178]:

## prepare data

train_path = "iwslt5k.en"
# train_path = ""

if train_path != "":
    
    training_data = []
    train_vocab = {}
    with codecs.open(train_path) as fp:
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
    

print (len(training_data))
print (training_data[0:3])


# In[175]:

# prune vocab based on vocab size

import operator
sorted_train_vocab = sorted(train_vocab.items(), key=operator.itemgetter(1), reverse=True)
pruned_vocab = sorted_train_vocab[0:VOCAB_SIZE]
pruned_vocab = ([i[0] for i in pruned_vocab])


# In[183]:

word2idx = {}
idx2word = {}
train = []
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


# In[184]:

print(train.shape)
print (train[0:13])


# In[192]:

# set input in the form of tensor
def prepare_input(batch):
    tensor_ids = torch.from_numpy(batch).to(device)
    return tensor_ids


# In[193]:

def batchify(train, max_sequence_length, batch_size, word2idx):
    
    batch_sequence_length = max_sequence_length*batch_size
    pad = np.array((batch_sequence_length - len(train)%(batch_sequence_length)) * [word2idx["eos"]], dtype=np.int)
    batches = np.concatenate((train, pad)).reshape((-1, BATCH_SIZE, MAX_SEQ_LENGTH))
    # batches is now [num_batches x sentences_in_minibatch x words_in_sentence]
    
    # torch expects words_in_sentence x sentences_in_minibatch
    # so, swap
    batches = np.swapaxes(batches, 1, 2)
    
    print(batches.shape)
    return batches

batches = batchify(train, MAX_SEQ_LENGTH, BATCH_SIZE, word2idx)


# In[187]:

#print(batches[0, :, 0])
#print(batches[0, :, 1])
print (batches[0,:-1,0])


# In[188]:

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# In[189]:

## LSTM Model

class LSTMLM(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size):
        super(LSTMLM, self).__init__()
        
        self.hidden_size = hidden_size   
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, vocab_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # set the dimenionaly of the hidden layer
        # parameters are: num_layers, minibatch_size, hidden_dim
        cell = autograd.Variable(torch.zeros(1, BATCH_SIZE, self.hidden_size))
        hid = autograd.Variable(torch.zeros(1, BATCH_SIZE, self.hidden_size))
        return (cell, hid)
    
    def forward(self, sequence):
        embeds = self.word_embeddings(sequence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        output_space = self.hidden2output(lstm_out)
        output_scores = F.log_softmax(output_space, dim=2)

        return output_scores


# In[190]:

# define loss, model and optimization
model = LSTMLM(EMBEDDING_SIZE, HIDDEN_SIZE, len(word2idx))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# In[191]:

## training
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm_notebook(batches, desc="Epoch %d/%d"%(epoch+1, epochs)):
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
    print ("Loss: ", total_loss/BATCH_SIZE)


# In[130]:

# test data
test_path = "iwslt1k.en"
#train_path = ""

if test_path != "":
    
    test_data = []
    with codecs.open(test_path) as fp:
        for line in fp:
            test_data.append(line.strip().split() + ["eos"])
else:        
    test_data = [
        ("The dog ate the apple eos".split()),
        ("Everybody read that book eos".split()),
        ("The cat ate the orange eos".split()),
        ("Everybody eat the apple eos".split())
    ]

print (len(test_data))
test = []
for sentence in test_data:
    for word in sentence:
        if word in word2idx:
            test.append(word2idx[word])
        else:
            test.append(word2idx["UNK"])
test = np.array(test, dtype=np.int)
test_batches = batchify(test, MAX_SEQ_LENGTH, BATCH_SIZE, word2idx)


# In[131]:

# perplexity on a test set

def evaluate(batch_in):
    total_loss = 0
    for batch in tqdm_notebook(batch_in):
        model.zero_grad()
        model.hidden = model.init_hidden()

        X = prepare_input(batch[:-1,:])
        y = prepare_input(batch[1:,:])

        output_scores = model(X)
        true_y = y.contiguous().view(-1, 1).squeeze()
        pred_y = output_scores.view(-1, len(word2idx))

        loss = loss_function(pred_y, true_y)
        total_loss += loss.data

    print ("Loss: ", total_loss/BATCH_SIZE )

evaluate(test_batches)


# In[ ]:



