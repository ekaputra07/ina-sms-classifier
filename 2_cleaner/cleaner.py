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

    # take words that has length > 2
    filtered = filter(lambda w: len(w) > 2, match)
    return ' '.join(filtered)


def clean(in_file, out_file):
    df = pd.read_csv(in_file)
    df = df[['label', 'message']]

    df['message'] = df['message'].apply(message_cleaner)

    # remove duplicates
    df.drop_duplicates(subset=['message'], inplace=True)
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset cleaner')
    parser.add_argument(
        '--csv-in',
        required=True,
        help='Input file to cleanup (CSV)'
    )
    parser.add_argument(
        '--csv-out',
        required=True,
        help='CSV Output file name'
    )
    args = parser.parse_args()

    input_file = args.csv_in
    output_file = args.csv_out

    clean(input_file, output_file)
