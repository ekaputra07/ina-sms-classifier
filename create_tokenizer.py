import argparse
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer

from utils.cleaner import basic_cleaner


def get_args():
    parser = argparse.ArgumentParser(
        description='Create tokenizer object file')

    parser.add_argument(
        '--input',
        required=True,
        help='Input file to read (must be CSV file)'
    )
    parser.add_argument(
        '--text-column',
        default='text',
        help='Name of the text column'
    )
    parser.add_argument(
        '--max-words',
        default=20000,
        help='Maximum number of words to use when tokenize sentences'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Where to store the tokenizer object'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    input_file = args.input
    output_file = args.output
    column = args.text_column
    max_words = args.max_words

    # read CSV input
    df = pd.read_csv(input_file)

    # do some basic cleaning
    df[column] = df[column].apply(basic_cleaner)

    # dedup
    df.drop_duplicates(subset=[column], inplace=True)

    # create tokenizer
    tokenizer = Tokenizer(num_words=int(max_words))
    tokenizer.fit_on_texts(df[column].values)

    with open(output_file, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    # how to load tokenizer
    # with open(output_file, 'rb') as f:
    #     t = pickle.load(f)
