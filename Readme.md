# ina-sms-classifier

A project to create Machine Learning model to classify Indonesian text/sms messages using [Tensorflow](https://www.tensorflow.org) and its [Keras](https://keras.io) api.

The main puspose is **to be able to detect scam/fraud SMS that often received by mobile phone users in Indonesia from unknown person and many have been reported to be victims of this kind of fraud activity**.

_Future plan_: the model can be transformed into [Tensorflow Lite](https://www.tensorflow.org/lite) and can be deployed as a mobile app that classify text message in real-time as it received by users. No need to send the message to model serving server to avoid privacy issue.

For now, it will classify messages into 4 classes:

- Scam (0)
- Online gambling website promotion (1)
- Online loans website promotion (2)
- Others (3)

Thanks to [laporsms.com](https://laporsms.com) for their effort collecting all the data that I've been using in this project.

## Usage

### Create text tokenizer
```
>> python create_tokenizer.py -h

usage: create_tokenizer.py [-h] --input INPUT [--text-column TEXT_COLUMN] [--max-words MAX_WORDS] --output OUTPUT

Create tokenizer object file

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input file to read (must be CSV file)
  --text-column TEXT_COLUMN
                        Name of the text column
  --max-words MAX_WORDS
                        Maximum number of words to use when tokenize sentences (default: 20000)
  --output OUTPUT       Where to store the tokenizer object
```

Example:
```
python create_tokenizer.py \
--input dataset/sms-row.csv \
--output model/tokenizer.pkl \
--text-column message
```

### Train and save the model

```
>> python create_model.py -h                                                                                                                                  

usage: create_model.py [-h] --tokenizer TOKENIZER --dataset DATASET [--text-column TEXT_COLUMN] [--label-column LABEL_COLUMN] [--max-words MAX_WORDS] [--maxlen MAXLEN] [--emb-dim EMB_DIM] [--class-num CLASS_NUM]
                       [--val-split VAL_SPLIT] [--test-split TEST_SPLIT] [--epochs EPOCHS] [--batch-size BATCH_SIZE] --output OUTPUT

Train and save model

optional arguments:
  -h, --help            show this help message and exit
  --tokenizer TOKENIZER
                        Path to saved tokenizer
  --dataset DATASET     Path to dataset file (must be CSV)
  --text-column TEXT_COLUMN
                        Name of the text column (default: text)
  --label-column LABEL_COLUMN
                        Name of the label column (default: label)
  --max-words MAX_WORDS
                        Max. number of words in vocabulary (must match tokenizer max-words, default: 20000)
  --maxlen MAXLEN       Max. number of words per message to use in training (default: 50)
  --emb-dim EMB_DIM     Words embedding dimension (default: 8)
  --class-num CLASS_NUM
                        Number of output classes (default: 4)
  --val-split VAL_SPLIT
                        Ratio of validation split (default: 0.2)
  --test-split TEST_SPLIT
                        Ratio of test split (default: 0.2)
  --epochs EPOCHS       Training epochs (default: 10)
  --batch-size BATCH_SIZE
                        Training batch size (default: 512)
  --output OUTPUT       Where to store the model
```

Example:
```
python create_model.py \
--tokenizer model/tokenizer.pkl \
--dataset dataset/sms-labeled-3k-clean.csv \
--text-column message \
--output model/latest
--epochs 75
```

At the end of the training you'll be asked whether you want to save the model, if yes then the model will be saved to `/model/latest`

### Model performance from latest training

*NOTE: below results are based on training 2700 of datapoints that are labeled from total of 18K (labeling all of them not finish yet).*
```
================== VALIDATION ===================
LOSS            : 0.13091
ACCURACY        : 0.94737
PRECISION       : 0.96234
RECALL          : 0.93117
AUC             : 0.99760


================== TEST ===================
LOSS            : 0.21565
ACCURACY        : 0.93091
PRECISION       : 0.94424
RECALL          : 0.92364
AUC             : 0.99164

CONFUSION MATRIX:
[[128   1   2   0]
 [  1  30   0   0]
 [  0   2  80   3]
 [  9   0   1  18]]

CLASSIFICATION REPORT:
              precision    recall  f1-score   support

           0       0.93      0.98      0.95       131
           1       0.91      0.97      0.94        31
           2       0.96      0.94      0.95        85
           3       0.86      0.64      0.73        28

    accuracy                           0.93       275
   macro avg       0.91      0.88      0.89       275
weighted avg       0.93      0.93      0.93       275
```

![Plot LOSS](https://github.com/ekaputra07/ina-sms-classifier/blob/master/plot_loss.png?raw=true)
![Plot ACC](https://github.com/ekaputra07/ina-sms-classifier/blob/master/plot_acc.png?raw=true)

### Development

I recommends you to install all the dependencies using [Conda]() and install the following libraries:
```
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
```

### License
```
Copyright (C) 2020  Eka Putra

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
```
