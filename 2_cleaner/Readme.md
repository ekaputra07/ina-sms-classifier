# Cleaner

`cleaner.py` is used to clean-up the dataset:

- remove duplicates
- remove punctuations
- remove unnecessary characters

Install dependencies:

```
pip install -r requirements.txt
```

To run the cleaner, run the following command:

```
python cleaner.py --jl-in=../dataset/sms-raw.jsonlines --csv-out=../dataset/sms-clean.csv
```
