from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import joblib
import pickle

data = pandas.read_csv('../dataset/sms.csv', encoding='utf-8', sep="|", na_values=["NULL"])
train_data = data[:49000]
test_data = data[49000:]

vectorizer = HashingVectorizer()
joblib.dump(vectorizer, '../model/vectorizer.pkl')

classifier = OneVsRestClassifier(SVC(kernel='linear'))

vectorize_text = vectorizer.fit_transform(train_data.message.astype('U'))
classifier.fit(vectorize_text, train_data.type.astype('U'))

with open('../model/model', 'wb') as f:
  pickle.dump(classifier, f)
