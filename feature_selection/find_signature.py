#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)

from sklearn import tree
from sklearn.metrics import accuracy_score

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
# words_file = "../text_learning/your_word_data.pkl" authors_file = "../text_learning/your_email_authors.pkl"
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]
print features_train



### your code goes here
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
# print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)
print len(features_train)
print acc
# print vectorizer.get_feature_names()[34597]
# importances = clf.feature_importances_
# import numpy as np
# indices = np.argsort(importances)[::-1]
# print 'Feature Ranking: '
# for i in range(10):
#     print "{} feature no.{} ({})".format(i+1,indices[i],importances[indices[i]])

print clf.feature_importances_.argmax()
print vectorizer.get_feature_names()[clf.feature_importances_.argmax()]

for counter, importance in enumerate(clf.feature_importances_):
    if importance > 0.2:
        print "Counter and importance ", counter, " - ", importance, " ", vectorizer.get_feature_names()[counter]
