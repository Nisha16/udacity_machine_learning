#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot
import numpy as np

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import f_regression

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import math
from sklearn.cross_validation import StratifiedShuffleSplit

def outliers_removal(data_dict):
    # Function removes the outliers from the dataset and returns the dictionary

    outliers = ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL']
    for outlier in outliers:
        data_dict.pop(outlier,0)
    return data_dict


def add_features(data_dict):

    ### Adding new featrues based on email numbers and largest valued columns.
    ratio_of_to_messages = []
    ratio_of_from_messages = []
    for user in data_dict:
        # print data_dict[user]['from_poi_to_this_person']
        if data_dict[user]['from_poi_to_this_person'] == "NaN" or data_dict[user]['to_messages'] == "NaN":
            data_dict[user]['ratio_of_to_messages'] = 0.

        else:
            data_dict[user]['ratio_of_to_messages'] = float(data_dict[user]['from_poi_to_this_person']) / float(data_dict[user]['to_messages'])
        if data_dict[user]['from_this_person_to_poi'] == "NaN" or data_dict[user]['from_messages'] == "NaN":
             data_dict[user]['ratio_of_from_messages'] = 0.
        else:
            data_dict[user]['ratio_of_from_messages'] = float(data_dict[user]['from_this_person_to_poi']) / float(data_dict[user]['from_messages'])
        if data_dict[user]['total_payments'] == "NaN":
            data_dict[user]['log_totalPayments'] = 0.
        elif data_dict[user]['total_payments'] > 0:
            data_dict[user]['log_totalPayments'] = math.log(data_dict[user]['total_payments'])
        else:
            data_dict[user]['log_totalPayments'] = data_dict[user]['total_payments']
        if data_dict[user]['total_stock_value'] == "NaN":
            data_dict[user]['log_total_stock_value'] = 0.
        elif data_dict[user]['total_stock_value'] > 0:
            data_dict[user]['log_total_stock_value'] = math.log(data_dict[user]['total_stock_value']+1)
        else:
            data_dict[user]['log_total_stock_value'] = data_dict[user]['total_stock_value']
    return data_dict

def duplicate_data(data_dict):
    count = 0
    for user in data_dict.keys():
        if data_dict[user]['poi']:
            count += 1
            data_dict[user+'dup'] = data_dict[user]
    print "POI's: ", count
    return data_dict



######################################################################################

def evaluate_classifiers(clf, features_test, labels_test):

    ### To calculate f1 score, recall and precision of each classifier
    labels_pred = clf.predict(features_test)
    # print "prediction: ", labels_pred

    f1 = f1_score(labels_test, labels_pred)
    print "f1 score: ", f1
    recall = recall_score(labels_test, labels_pred)
    print "recall: ", recall
    precision = precision_score(labels_test, labels_pred)
    print "precision: ", precision
    accuracy = accuracy_score(labels_test, labels_pred)
    print "accuracy: ", accuracy
    return f1, recall, precision, accuracy


def DecisionTree_clf(parameter_tuning, features_train, labels_train, features_test, labels_test):
    if parameter_tuning:
        t0 = time()
        estimators = [('select_best',SelectKBest()),('DT', DecisionTreeClassifier())]
        # estimators = [('pca',PCA()),('DT', DecisionTreeClassifier())]
        pipe = Pipeline(estimators)
        params = dict(select_best__k=[3,5,7], DT__min_samples_split=[3, 5,10,15,25,40], DT__max_features=['auto','sqrt'], DT__criterion=['gini', 'entropy'])
        clf = GridSearchCV(pipe, param_grid=params)
        # clf.fit(features_train, labels_train)
        # print "training time:", round(time()-t0, 3), "s"
        # print "Best parameters for decistion tree: ", clf.best_params_
        # print "Below are the f1 score, precision, recall & accuracy of decision tree classifier: "
        # f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)
    else:
        t0 = time()
        clf = DecisionTreeClassifier(min_samples_split=15)
        clf = clf.fit(features_train, labels_train)
        print "training time:", round(time()-t0, 3), "s"
        print "Below are the f1 score, precision, recall & accuracy of decision tree classifier: "
        f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)
    return clf

def randomForest_clf(parameter_tuning, pca, kBest, features_train, labels_train, features_test, labels_test):
    # create feature union
    if parameter_tuning:
        if kBest:
            print "Inside kbest random"
            t0 = time()
            estimators = [('select_best',SelectKBest()),('random', RandomForestClassifier())]
            pipe = Pipeline(estimators)
            params = dict(select_best__k=[5, 8, 10],random__n_estimators=[6, 7, 10, 15],random__criterion=['entropy'])
            clf = GridSearchCV(pipe, param_grid=params)
            # clf.fit(features_train, labels_train)
            # print "training time:", round(time()-t0, 3), "s"
            # print "Best parameters for random with kbest: ", clf.best_params_
            # print "Below are the f1 score, precision, recall & accuracy of Random forest classifier: "
            # f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)
            return clf
        if pca:
            print "Inside PCA random"
            t0 = time()
            estimators = [('reduce_dim', PCA()), ('random', RandomForestClassifier())]
            pipe = Pipeline(estimators)
            params = dict(reduce_dim__n_components=[ 4, 5],random__n_estimators=[10, 20, 30],random__criterion=['entropy'])
            clf = GridSearchCV(pipe, param_grid=params)
            # clf.fit(features_train, labels_train)
            # print "training time:", round(time()-t0, 3), "s"
            # print "Best parameters for random with PCA: ", clf.best_params_
            # print "Below are the f1 score, precision, recall & accuracy of Random forest classifier: "
            # f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)
            return clf
    else:
        t0 = time()
        clf = RandomForestClassifier(n_estimators=2,criterion="entropy")
        clf.fit(features_train, labels_train)
        print "training time:", round(time()-t0, 3), "s"
        print "Below are the f1 score, precision, recall & accuracy of random forest classifier: "
        f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)
        return clf

def knn_clf(parameter_tuning, pca, kBest, features_train, labels_train, features_test, labels_test):
    if parameter_tuning:
        if pca:
            print "Inside PCA KNN"
            t0 = time()
            estimators = [('reduce_dim', PCA()), ('knn', KNeighborsClassifier())]
            pipe = Pipeline(estimators)
            params = dict(reduce_dim__n_components=[4, 5, 10],knn__n_neighbors=[3, 4, 5, 6, 7],knn__weights=['uniform','distance'])
            clf = GridSearchCV(pipe, param_grid=params)
            # clf.fit(features_train, labels_train)
            # print "training time:", round(time()-t0, 3), "s"
            # print "Best parameters for KNN with PCA: ", clf.best_params_
            # print "Below are the f1 score, precision, recall & accuracy of KNN classifier: "
            # f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)

        if kBest:
            print "Inside Kbest KNN"
            t0 = time()
            estimators = [('select_best',SelectKBest()),('knn', KNeighborsClassifier())]
            pipe = Pipeline(estimators)
            params = dict(select_best__k=[5, 10, 15 ],knn__n_neighbors=[3, 4, 5, 6, 7], knn__weights=['uniform','distance'])
            clf = GridSearchCV(pipe, param_grid=params)
            # clf.fit(features_train, labels_train)
            # print "training time:", round(time()-t0, 3), "s"
            # print "Best parameters for KNN with SelectKBest: ", clf.best_params_
            # print "Below are the f1 score, precision, recall & accuracy of KNN classifier: "
            # f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)
    else:
        t0 = time()
        clf = KNeighborsClassifier(n_neighbors=4)
        clf.fit(features_train, labels_train)
        print "training time:", round(time()-t0, 3), "s"
        print "Below are the f1 score, precision, recall & accuracy of decision tree classifier: "
        f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)
    return clf

def gaussian_clf(parameter_tuning, features_train, labels_train, features_test, labels_test):
    if parameter_tuning:
        t0 = time()
        estimators = [('select_best',SelectKBest()),('gaussian', GaussianNB())]
        # estimators = [('pca',PCA()),('gaussian', GaussianNB())]
        pipe = Pipeline(estimators)
        params = dict(select_best__k=[3,5,7,10,15])
        clf = GridSearchCV(pipe, param_grid=params)
        # clf.fit(features_train, labels_train)
        # print "training time:", round(time()-t0, 3), "s"
        # print "Best parameters for gaussian: ", clf.best_params_
        # print "Below are the f1 score, precision, recall & accuracy of gaussian classifier: "
        # f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)
        return pipe
    else:
        t0 = time()
        clf = GaussianNB()
        clf.fit(features_train, labels_train)
        print "training time:", round(time()-t0, 3), "s"
        print "Below are the f1 score, precision, recall & accuracy of gaussian classifier: "
        f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)
        return clf

def Evaluate_clf_randomSampling(decisionTree, knn, random, gaussian):

    poi = ['poi']

    ###all the email related features which holds value as numbers except  ,
    email_features = [
                # "from_messages",
                "from_poi_to_this_person",
                "from_this_person_to_poi",
                "shared_receipt_with_poi",
                # "to_messages"
    ]

    ### financial data features ommitted 'loan advances', 'director_fees'
    financial_features = [
                "bonus",
                "deferral_payments",
                "deferred_income",
                "exercised_stock_options",
                "expenses",
                "long_term_incentive",
                # "other",
                "restricted_stock",
                "restricted_stock_deferred",
                "salary"]
                # "total_payments",
                # "total_stock_value"]

    new_features = ['ratio_of_from_messages','ratio_of_to_messages','log_totalPayments','log_total_stock_value']


    features_list =  poi + email_features + financial_features + new_features# You will need to use more features poi + email_features + financial_features
    # print "FEature list: ", features_list

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # To remove outlier and add new features
    data_dict = outliers_removal(data_dict)
    print "Removing outliers and adding new features"
    # data_dict = add_features(data_dict, financial_features)
    data_dict = add_features(data_dict)
    data_dict = duplicate_data(data_dict)
    print "length of dataset: ", len(data_dict)

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    # print "Dataset: ", my_dataset

    ## Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    pca = False
    kBest = True
    parameter_tuning = True
    if decisionTree:
        clf = DecisionTree_clf(parameter_tuning, features_train, labels_train, features_test, labels_test)
    elif knn:
        clf = knn_clf(parameter_tuning, pca, kBest, features_train, labels_train, features_test, labels_test)
    elif random:
        clf = randomForest_clf(parameter_tuning, pca, kBest, features_train, labels_train, features_test, labels_test)
    elif gaussian:
        clf = gaussian_clf(parameter_tuning, features_train, labels_train, features_test, labels_test)

    print "Dumping data"
    dump_classifier_and_data(clf, my_dataset, features_list)

    return None

def Evaluate_clf_kfold(decisionTree, knn, random, gaussian):
    poi = ['poi']

    ###all the email related features which holds value as numbers except  ,
    email_features = [
                # "from_messages",
                "from_poi_to_this_person",
                "from_this_person_to_poi",
                "shared_receipt_with_poi",
                # "to_messages"
    ]

    ### financial data features ommitted 'loan advances', 'director_fees'
    financial_features = [
                "bonus",
                "deferral_payments",
                "deferred_income",
                "exercised_stock_options",
                "expenses",
                "long_term_incentive",
                # "other",
                "restricted_stock",
                "restricted_stock_deferred",
                "salary"]
                # "total_payments",
                # "total_stock_value"]

    new_features = ['ratio_of_from_messages','ratio_of_to_messages','log_totalPayments','log_total_stock_value']


    features_list =  poi + email_features + financial_features + new_features# You will need to use more features poi + email_features + financial_features
    # print "FEature list: ", features_list

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # To remove outlier and add new features
    data_dict = outliers_removal(data_dict)
    print "Removing outliers and adding new features"
    # data_dict = add_features(data_dict, financial_features)
    data_dict = add_features(data_dict)
    data_dict = duplicate_data(data_dict)
    print "length of dataset: ", len(data_dict)

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    # print "Dataset: ", my_dataset

    ## Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    folds = 1000
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

    pca = False
    kBest = True
    parameter_tuning = True
    if decisionTree:
        clf = DecisionTree_clf(parameter_tuning, features_train, labels_train, features_test, labels_test)
    elif knn:
        clf = knn_clf(parameter_tuning, pca, kBest, features_train, labels_train, features_test, labels_test)
    elif random:
        clf = randomForest_clf(parameter_tuning, pca, kBest, features_train, labels_train, features_test, labels_test)
    elif gaussian:
        clf = gaussian_clf(parameter_tuning, features_train, labels_train, features_test, labels_test)


    print "Dumping data"
    dump_classifier_and_data(clf, my_dataset, features_list)

    return None

decisionTree, knn, random, gaussian = False, False, False, True

Evaluate_clf_randomSampling(decisionTree, knn, random, gaussian)

# Evaluate_clf_kfold(decisionTree, knn, random, gaussian)
