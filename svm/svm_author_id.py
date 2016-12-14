#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 



#########################################################
### your code goes here ###
# C = [1.0,10.0,100.0,1000.0,10000.0]
# accuracy = []
# for c in C:
clf = svm.SVC(C=10000.0,kernel="rbf")
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
l = list(prediction)
print "prediction: ", l.count(1)
# acc = accuracy_score(labels_test, prediction)
# accuracy.append(acc)
# print acc

#########################################################


