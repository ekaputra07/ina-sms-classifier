import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

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
    parser.add_argument(
        '--test',
        action='store_true',
        help='Whether to evaluate the test set, otherwise it will only run training and show validation metrics'
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
        layers.Embedding(max_words, emb_dim, input_length=maxlen),
        layers.Flatten(),
        layers.Dense(output, activation='softmax')  # 4 probability output
    ])
    metrics = [
        CategoricalAccuracy(),
        Precision(),
        Recall(),
        AUC()
    ]
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=metrics)
    return model


def render_metrics(epochs, history):
    """
    Plot training metrics and save to file
    - plot_loss.png
    - plot_acc.png
    """
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['categorical_accuracy']
    val_acc = history['val_categorical_accuracy']

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


def show_evaluation_result(step, result):
    """
    Show result of model.evaluate()
    """
    print('\n\n================== {} ==================='.format(step))
    print('LOSS\t\t: {:.5f}'.format(result[0]))
    print('ACCURACY\t: {:.5f}'.format(result[1]))
    print('PRECISION\t: {:.5f}'.format(result[2]))
    print('RECALL\t\t: {:.5f}'.format(result[3]))
    print('AUC\t\t: {:.5f}'.format(result[4]))


def show_classification_report(y_true, y_pred):
    """
    Show confusion matrix and classification report
    """
    print('\nCONFUSION MATRIX:')
    print(confusion_matrix(y_true=y_true,
                           y_pred=y_pred))

    print('\nCLASSIFICATION REPORT:')
    print(classification_report(y_true=y_true, y_pred=y_pred))


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

    texts = shuffled[args.text_column].values
    labels = to_categorical(shuffled[args.label_column].values)

    # tokenize texts
    tokens = tokenizer.texts_to_sequences(texts)

    # pad sequences
    data = pad_sequences(tokens, maxlen=args.maxlen)

    # split train & test data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=args.test_split, random_state=85)

    # get a portion of validation data from training data
    X_train2, X_val, y_train2, y_val = train_test_split(
        X_train, y_train, test_size=args.val_split, random_state=85)

    # create model
    model = get_model(args.max_words, args.maxlen,
                      emb_dim=args.emb_dim,
                      output=args.class_num)

    # train
    hist = model.fit(X_train2, y_train2,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     validation_data=(X_val, y_val),
                     verbose=0)

    # display metrics
    render_metrics(args.epochs, hist.history)

    # evaluate using validation data
    val_result = model.evaluate(X_val, y_val)
    if not args.test:
        show_evaluation_result('VALIDATION', val_result)

    else:
        # Create new model and train using the whole training dataset
        model2 = get_model(args.max_words, args.maxlen,
                           emb_dim=args.emb_dim,
                           output=args.class_num)
        model2.fit(X_train, y_train,
                   epochs=args.epochs,
                   batch_size=args.batch_size,
                   verbose=0)

        # evaluate model
        test_result = model2.evaluate(X_test, y_test)

        # show both training and test result so we can compare them
        show_evaluation_result('VALIDATION', val_result)
        show_evaluation_result('TEST', test_result)

        predictions = model2.predict(X_test)
        show_classification_report(y_true=np.argmax(y_test, axis=1),
                                   y_pred=np.argmax(predictions, axis=1))

        # save the model?
        save_model = input("Save model to disk? (y/[N]): ")
        if save_model.lower() == 'y':
            model2.save(args.output)
            print('Model saved to {}'.format(args.output))
