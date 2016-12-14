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
from sklearn.cluster import KMeans
from sklearn.svm import SVC

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Dividing feature list into different categories
poi = ['poi']

###all the email related features which holds value as numbers
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### financial data features ommitted 'loan advances', 'director_fees','restricted_stock_deferred',
financial_features = ['salary', 'deferral_payments', 'total_payments', 'bonus',  'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock']

###email addresses
email_address = ['email_address']
features_list =  poi + email_features + financial_features# You will need to use more features poi + email_features + financial_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
def outliers_removal():
    # Function removes the outliers from the dataset and returns the dictionary

    outliers = ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL', 'LOCKHART EUGENE E']
    for outlier in outliers:
        data_dict.pop(outlier,0)
    return data_dict

### Task 3: Create new feature(s)
def add_features(data_dict):

    ### Adding new featrues based on email numbers and largest valued columns.
    ratio_of_to_messages = []
    ratio_of_from_messages = []
    for user in data_dict:
        try:
            data_dict[user]['ratio_of_to_messages'] = float(data_dict[user]['from_poi_to_this_person']) / data_dict[user]['to_messages']
            data_dict[user]['ratio_of_from_messages'] = float(data_dict[user]['from_this_person_to_poi']) / data_dict[user]['from_messages']
            data_dict[user]['log_totalPayments'] = math.log10(data_dict[name]['total_payments'])
            data_dict[user]['log_total_stock_value'] = math.log10(data_dict[name]['total_stock_value'])
        except:
            data_dict[user]['ratio_of_to_messages'] = "NaN"
            data_dict[user]['ratio_of_from_messages'] = "NaN"
            data_dict[user]['log_totalPayments'] = "NaN"
            data_dict[user]['log_total_stock_value'] = "NaN"
    return data_dict

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

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


def DecisionTree_clf():
    t0 = time()
    parameters = {'min_samples_split':[5,10,15,25,40], 'max_features':('auto','sqrt'), 'criterion':('gini', 'entropy')}
    tree = DecisionTreeClassifier()
    clf = GridSearchCV(tree, parameters)
    clf.fit(features_train, labels_train)
    # print "training time:", round(time()-t0, 3), "s"
    print "Best parameters for decistion tree: ", clf.best_params_
    print "Below are the f1 score, precision, recall & accuracy of classifier: "
    f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)
    return f1_tree, recall_tree, precision_tree, accuracy

def kmeans_clf():
    estimators = [('reduce_dim', PCA()), ('kmeans', KMeans())]
    pipe = Pipeline(estimators)
    params = dict(reduce_dim__n_components=[2, 5, 6],kmeans__n_clusters=[2],kmeans__n_init=[15])
    clf = GridSearchCV(pipe, param_grid=params)
    pred = clf.fit_predict( features )
    print "Best parameters for SVC with PCA: ", grid_search.best_params_
    try:
        Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
    except NameError:
        print "no predictions object named pred found, no clusters to plot"
    return None

def SVC_clf(pca, kBest):
    # create feature union
    best_feat = SelectKBest(f_regression, k=5)
    clf = GaussianNB()
    pipe = Pipeline([('kbest', best_feat), ('gaus', clf)])
    pipe.set_params(kbest__k=10).fit(features_train, labels_train)
    print "Below are the f1 score, precision, recall & accuracy of gaussian classifier: "
    f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(pipe, features_test, labels_test)
    return f1_tree, recall_tree, precision_tree, accuracy

    if pca:
        estimators = [('reduce_dim', PCA()), ('clf', SVC())]
        pipe = Pipeline(estimators)
        params = dict(reduce_dim__n_components=[2, 5, 10],clf__C=[0.1, 10, 100])
        grid_search = GridSearchCV(pipe, param_grid=params)
        grid_search.fit(features_train, labels_train)
        print "Best parameters for SVC with PCA: ", grid_search.best_params_
        print "Below are the f1 score, precision, recall & accuracy of classifier: "
        f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(grid_search, features_test, labels_test)

    if kBest:
        estimators = [('select_best',SelectKBest()),('clf', SVC())]
        pipe = Pipeline(estimators)
        params = dict(select_best__k=[2, 5, 10],clf__C=[0.1, 10, 100])
        grid_search = GridSearchCV(pipe, param_grid=params)
        grid_search.fit(features_train, labels_train)
        print "Best parameters for SVC with SelectKBest: ", grid_search.best_params_
        print "Below are the f1 score, precision, recall & accuracy of classifier: "
        f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(grid_search, features_test, labels_test)

    return f1_tree, recall_tree, precision_tree, accuracy

def knn_clf(pca, kBest):
    if pca:
        estimators = [('reduce_dim', PCA()), ('knn', KNeighborsClassifier())]
        pipe = Pipeline(estimators)
        params = dict(reduce_dim__n_components=[2, 5, 8, 10],knn__n_neighbors=[3, 4, 5],knn__weights=['uniform','distance'])
        clf = GridSearchCV(pipe, param_grid=params)
        clf.fit(features_train, labels_train)
        print "Best parameters for SVC with PCA: ", clf.best_params_
        print "Below are the f1 score, precision, recall & accuracy of classifier: "
        f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)

    if kBest:
        estimators = [('select_best',SelectKBest()),('knn', KNeighborsClassifier())]
        pipe = Pipeline(estimators)
        params = dict(select_best__k=[2, 5, 10],knn__C=[3, 4, 5],knn__weights=['uniform','distance'])
        clf = GridSearchCV(pipe, param_grid=params)
        clf.fit(features_train, labels_train)
        print "Best parameters for SVC with SelectKBest: ", clf.best_params_
        print "Below are the f1 score, precision, recall & accuracy of classifier: "
        f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(clf, features_test, labels_test)

    return f1_tree, recall_tree, precision_tree, accuracy

def gaussian_clf():
    best_feat = SelectKBest(f_regression, k=5)
    clf = GaussianNB()
    pipe = Pipeline([('kbest', best_feat), ('gaus', clf)])
    pipe.set_params(kbest__k=10).fit(features_train, labels_train)
    print "Below are the f1 score, precision, recall & accuracy of gaussian classifier: "
    f1_tree, recall_tree, precision_tree, accuracy = evaluate_classifiers(pipe, features_test, labels_test)
    return f1_tree, recall_tree, precision_tree, accuracy

def randomforest_clf():
    return

def logistic_clf():
    return





print "Calling outliers_removal"
outliers_removal()
print "Calling add_features"
add_features(data_dict)



print "length of dataset: ", len(data_dict)

# Adding new features after removing the outliers
data_dict = add_features(outliers_removal())

### Store to my_dataset for easy export below.
my_dataset = data_dict

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=True, remove_all_zeroes=True)

print "print the data: ", data
labels, features = targetFeatureSplit(data)




### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

DecisionTree_clf()
# SVC_clf(pca=True,kBest=False)
gaussian_clf()
knn_clf(pca=True,kBest=False)
kmeans_clf()





### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
