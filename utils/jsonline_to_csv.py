import pandas as pd
import argparse


def convert(in_file, out_file):
    df = pd.read_json(in_file, lines=True)
    df = df[['type', 'message', 'sender']]
    df['message'] = df['message'].apply(
        lambda m: " ".join(m.strip().lower().split()))
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

    convert(input_file, output_file)
