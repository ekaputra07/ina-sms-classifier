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


def perform(classifiers, vectorizers, train_data, test_data):
    for classifier in classifiers:
      for vectorizer in vectorizers:
        string = ''
        string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

        # train
        vectorize_text = vectorizer.fit_transform(train_data.message.astype('U'))
        classifier.fit(vectorize_text, train_data.type.astype('U'))

        # score
        vectorize_text = vectorizer.transform(test_data.message.astype('U'))
        score = classifier.score(vectorize_text, test_data.type.astype('U'))
        string += '. Has score: ' + str(score)
        print(string)


data = pandas.read_csv('../dataset/sms.csv', sep="|", na_values=["NULL"])
learn = data[:49000] # 4400 items
test = data[49000:] # 1172 items

perform(
    [
        # BernoulliNB(),
        # RandomForestClassifier(n_estimators=100, n_jobs=-1),
        # AdaBoostClassifier(),
        # BaggingClassifier(),
        # ExtraTreesClassifier(),
        # GradientBoostingClassifier(),
        # DecisionTreeClassifier(),
        # CalibratedClassifierCV(),
        # DummyClassifier(),
        # PassiveAggressiveClassifier(),
        # RidgeClassifier(),
        # RidgeClassifierCV(),
        # SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        KNeighborsClassifier()
    ],
    [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer()
    ],
    learn,
    test
)

