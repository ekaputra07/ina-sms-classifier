from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

data = pandas.read_csv('../dataset/sms.csv', encoding='utf-8', sep="|", na_values=["NULL"])
train_data = data[:49000]
test_data = data[49000:]

classifier = OneVsRestClassifier(SVC(kernel='linear'))
# vectorizer = TfidfVectorizer()
vectorizer = HashingVectorizer()

# train
vectorize_text = vectorizer.fit_transform(train_data.message.astype('U'))
classifier.fit(vectorize_text, train_data.type.astype('U'))

vectorize_text = vectorizer.transform(test_data.message.astype('U'))
score = classifier.score(vectorize_text, test_data.type.astype('U'))
print(score) # 98,8
