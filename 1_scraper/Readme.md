# Scaper

This is a web scraper/crawler to extract data from http://laporsms.com.

Install dependencies:

```
pip install -r requirements.txt
```

To run the crawler, the result is formatted as Json Lines:

```
scrapy crawl laporan_masyarakat -o ../../dataset/sms.jl
```
