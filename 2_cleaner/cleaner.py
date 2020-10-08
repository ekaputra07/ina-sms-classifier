import re
import string
import pandas as pd
import argparse


def message_cleaner(message):
    """
    Do some basic cleaning
    """
    # replace with space
    message = message\
        .strip()\
        .lower()\
        .replace('\n', ' ')

    # get all words (ignore number)
    match = re.findall("[a-zA-Z]+", message)

    # take words that has length > 3
    filtered = filter(lambda w: len(w) > 3, match)
    return ' '.join(filtered)


def clean(in_file, out_file):
    df = pd.read_json(in_file, lines=True)
    df = df[['message', 'type']]

    # convert type to 1 and 0
    # Penipuan = 1, else = 0
    # NOTE: this is not final label, many of the label are wrong, need to verify manually before used for training.
    df['type'] = df['type'].apply(lambda t: 1 if t == 'Penipuan' else 0)
    df['message'] = df['message'].apply(message_cleaner)

    # remove duplicates
    df.drop_duplicates(subset=['message'], inplace=True)
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset cleaner')
    parser.add_argument(
        '--jl-in',
        required=True,
        help='Input file to cleanup (must be Jsonline from scrapper)'
    )
    parser.add_argument(
        '--csv-out',
        required=True,
        help='CSV Output file name'
    )
    args = parser.parse_args()

    input_file = args.jl_in
    output_file = args.csv_out

    clean(input_file, output_file)
