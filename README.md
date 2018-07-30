# lstm-lm
LSTM based language model in Pytorch

### Preprocess the data
```
python preprocess.py --train train_data --validation valid_data --test test_data --output_dir .
```

### Train the model
```
python lstm_lm.py  --data_dir . --epochs 2 --dropout 0.3 --rnn_layers 2 --output_model sample_model
```

### Evaluate the model
```
python evaluate.py --model sample_model_1 --vocab vocab.json --test iwslt1k.en
```

Extensions:
* char-based RNN
* char-based CNN