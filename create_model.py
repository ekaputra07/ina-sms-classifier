import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

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
        type=int,
        default=20000,
        help='Max. number of words in vocabulary (must match tokenizer max-words, default: 20000)'
    )
    parser.add_argument(
        '--maxlen',
        type=int,
        default=50,
        help='Max. number of words per message to use in training (default: 50)'
    )
    parser.add_argument(
        '--emb-dim',
        type=int,
        default=8,
        help='Words embedding dimension (default: 8)'
    )
    parser.add_argument(
        '--class-num',
        type=int,
        default=4,
        help='Number of output classes (default: 4)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Ratio of validation split (default: 0.2)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Ratio of test split (default: 0.2)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Training batch size (default: 512)'
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


def train_test_split(data, labels, test_ratio):
    """
    Split train and test dataset.
    """
    total = data.shape[0]
    test_max_index = int(total * test_ratio)
    return (data[test_max_index:], labels[test_max_index:]), (data[:test_max_index], labels[:test_max_index])


def get_model(max_words, maxlen, emb_dim=8, output=4):
    """
    Returns multiclass classification model
    """
    model = models.Sequential([
        layers.Embedding(max_words, emb_dim, input_length=maxlen),
        layers.Flatten(),
        layers.Dense(output, activation='softmax')  # 4 probability output
    ])
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model


def render_metrics(epochs, history):
    """
    Plot training metrics and save to file
    - plot_loss.png
    - plot_acc.png
    """
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['acc']
    val_acc = history['val_acc']

    plt.plot(range(1, epochs+1), train_loss, 'b', label='Training loss')
    plt.plot(range(1, epochs+1), val_loss, 'bo', label='Validation loss')
    plt.xlabel('epochs ({})'.format(epochs))
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('plot_loss.png')
    plt.clf()

    plt.plot(range(1, epochs+1), train_acc, 'b', label='Training acc')
    plt.plot(range(1, epochs+1), val_acc, 'bo', label='Validation acc')
    plt.xlabel('epochs ({})'.format(epochs))
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('plot_acc.png')


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

    # pad sequences
    data = pad_sequences(tokens, maxlen=args.maxlen)

    # split raining & test data
    (x_train, y_train), (x_test, y_test) = train_test_split(
        data, labels, args.test_split)

    # create model
    model = get_model(args.max_words, args.maxlen,
                      emb_dim=args.emb_dim, output=args.class_num)

    # train
    hist = model.fit(x_train, y_train, epochs=args.epochs,
                     batch_size=args.batch_size, validation_split=args.val_split)

    # display metrics
    render_metrics(args.epochs, hist.history)

    # evaluate model
    result = model.evaluate(x_test, y_test)
    print('=====================================')
    print('LOSS: {:.2f}'.format(result[0]))
    print('ACCURACY: {:.2f}'.format(result[1]))
    print('=====================================')

    # save the model?
    save_model = input("Save model to disk? [y/N]: ")
    if save_model.lower() == 'y':
        model.save(args.output)
        print('Model saved to {}'.format(args.output))
