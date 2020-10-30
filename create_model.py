import argparse
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.cleaner import basic_cleaner


def get_args():
    """
    Read program arguments
    """
    parser = argparse.ArgumentParser(
        description='Train and save model')

    parser.add_argument(
        '--tokenizer',
        required=True,
        help='Path to saved tokenizer'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to dataset file (must be CSV)'
    )
    parser.add_argument(
        '--text-column',
        default='text',
        help='Name of the text column (default: text)'
    )
    parser.add_argument(
        '--label-column',
        default='label',
        help='Name of the label column (default: label)'
    )
    parser.add_argument(
        '--max-words',
        default=20000,
        help='Max. number of words in vocabulary (must match tokenizer max-words, default: 20000)'
    )
    parser.add_argument(
        '--maxlen',
        default=50,
        help='Max. number of words per message to use in training (default: 50)'
    )
    parser.add_argument(
        '--emb-dim',
        default=8,
        help='Words embedding dimension (default: 8)'
    )
    parser.add_argument(
        '--class-num',
        default=4,
        help='Number of output classes (default: 4)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Where to store the model'
    )
    return parser.parse_args()


def get_tokenizer(path):
    """
    Load tokenizer from file
    """
    tokenizer = None
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def get_model(max_words, maxlen, emb_dim=8, output=4):
    """
    Returns multiclass classification model
    """
    model = models.Sequential([
        # 8 embedding dimension
        layers.Embedding(max_words, emb_dim, input_length=maxlen),
        layers.Flatten(),
        layers.Dense(output, activation='softmax')  # 4 probability output
    ])
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model


if __name__ == '__main__':
    args = get_args()

    # load and do some cleaning
    dataset = pd.read_csv(args.dataset)
    dataset[args.text_column] = dataset[args.text_column].apply(basic_cleaner)

    # remove duplication
    dataset.drop_duplicates(subset=[args.text_column], inplace=True)

    # shuffles dataset
    shuffled = dataset.sample(frac=1).reset_index(drop=True)

    # load tokenizer
    tokenizer = get_tokenizer(args.tokenizer)

    labels = shuffled[args.label_column].values
    texts = shuffled[args.text_column].values

    # tokenize texts
    tokens = tokenizer.texts_to_sequences(texts)

    # pas sequences
    data = pad_sequences(tokens, maxlen=int(args.maxlen))

    # TODO: split raining & test data

    # create model
    model = get_model(int(args.max_words), int(args.maxlen),
                      emb_dim=int(args.emb_dim), output=int(args.class_num))

    # TODO: train

    # TODO: show metrics

    # TODO: save model
