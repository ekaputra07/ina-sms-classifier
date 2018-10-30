from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import csv

data = pandas.read_csv('../dataset/sms.csv', encoding='utf-8', sep="|", na_values=["NULL"])
train_data = data[:30376]
test_data = data[30376:]

classifier = OneVsRestClassifier(SVC(kernel='linear'))
#vectorizer = TfidfVectorizer()
#vectorizer = HashingVectorizer()
vectorizer = CountVectorizer()

# train
vectorize_text = vectorizer.fit_transform(train_data.message.astype('U'))
classifier.fit(vectorize_text, train_data.type.astype('U'))

# score
# vectorize_text = vectorizer.transform(test_data.v2)
# score = classifier.score(vectorize_text, test_data.v1)
# print(score) # 98,8


csv_arr = []
for index, row in test_data.iterrows():
    answer = row[1]
    text = row[3]
    vectorize_text = vectorizer.transform([text])
    predict = classifier.predict(vectorize_text)[0]
    if predict == answer:
        result = '_RIGHT'
    else:
        result = '_WRONG'
    csv_arr.append([len(csv_arr), text, answer, predict, result])


# write csv
with open('../dataset/test_score.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='|',
            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['#', 'text', 'answer', 'predict', result])

    for row in csv_arr:
        spamwriter.writerow(row)
