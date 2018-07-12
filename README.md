# lstm-lm
LSTM based language model in Pytorch

### Preprocess the data
```
python preprocess.py --train train_data --validation valid_data --test test_data --output_dir .
```

### Train the model
```
python lstm-lm.py  --data_dir . --epochs 2 --dropout 0.3 --rnn_layers 2 --output_model model_iwslt5
```

### Evaluate the model (work in progress)
```
python evaluate.py --model model_iwslt5_1 --vocab vocab.json --test iwslt1k.en
```

TODO:
* Model saving with paramters
* evaluation script


Extensions:
* char-based RNN
* char-based CNN