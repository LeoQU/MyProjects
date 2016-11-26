#!/usr/bin/python

import sys

import pprint
pp = pprint.PrettyPrinter(indent=4)

import csv
import pickle
import numpy as np
from time import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


with open("enron_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# fix Sanjay Bhatnagar's data
data_dict['BHATNAGAR SANJAY']['director_fees'] = 0
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['other'] = 0
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

# fix Robert Belfer's data
data_dict['BELFER ROBERT']['deferral_payments'] = 0
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['exercised_stock_options'] = 0
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['total_stock_value'] = 0

### Task 1: Identify outliers
# The details of identifying outliers are presented in identify_outliers.html
non_employee = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
outliers_1 = ['FREVERT MARK A', 'ALLEN PHILLIP K']
outliers_2 = ['BECK SALLY W', 'KITCHEN LOUISE', 'PAI LOU L', 'SHAPIRO RICHARD S', 'URQUHART JOHN A']

for i in non_employee + outliers_1:
    data_dict.pop(i) # remove 4 outliers

### Task 2: Design new features
# list all the numeric features
all_features = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi', 'to_messages', 'total_payments', 'total_stock_value']

# introduce two new features
new_features = ['NaN_num', 'poi_message_over_total_message']

# return the number of 'NaN' in a dictionary
def count_NaN(dic):
    count = 0
    for value in dic.values():
        if value == 'NaN':
            count += 1
    return count

# insert new features into dataset
if new_features != []:
    for value in data_dict.values():
        value[new_features[0]] = count_NaN(value)
        if value['from_poi_to_this_person'] != 'NaN' and value['from_this_person_to_poi'] != 'NaN' and value['from_messages'] != 'NaN' and value['to_messages'] != 'NaN':
            value[new_features[1]] = ( float(value['from_poi_to_this_person']) + float(value['from_this_person_to_poi']) ) / ( float(value['from_messages']) + float(value['to_messages']) )
        else: value[new_features[1]] = 'NaN'

### Task 3: Feature selection
data = featureFormat(data_dict, all_features+new_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

# use RFECV for feature selection
print 'All features:', all_features+new_features
print ''
from sklearn import tree
clf = tree.DecisionTreeClassifier() # use decision tree for feature selection

from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=clf, step=1, cv=4)
rfecv.fit(features, labels)

tmp_list = all_features+new_features
tmp_list.remove('poi')
features_list = ['poi'] + list( np.array(tmp_list)[rfecv.support_] )

print 'Selected features by REFCV:', features_list
print 'The rank of all features:', rfecv.ranking_
print 'The score of every subset:', rfecv.grid_scores_
print ''

# use univariate feature selection
from sklearn.feature_selection import SelectPercentile
selector = SelectPercentile(percentile=30).fit(features, labels)

tmp_list = all_features+new_features
tmp_list.remove('poi')
features_list = ['poi'] + list( np.array(tmp_list)[selector.get_support()] )

print 'Selected features by SelectPercentile:', features_list
print 'Feature scores:', selector.scores_
print ''

# best features
features_list = ['poi', 'bonus', 'deferred_income', 'expenses', 'total_payments', 'total_stock_value'] + new_features
 
### Task 4&5: Try a varity of classifiers and Tune classifier parameters
# Extract features and labels from the dataset
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Feature scaling
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
features_scaled = min_max_scaler.fit_transform(features)

# split train set and test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2) # set 3 principle components
pca.fit(features_train)
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)

# tune classifier parameters
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


classifiers = [GaussianNB, svm.SVC,tree.DecisionTreeClassifier]
parameters = [{}, 
              {'kernel':('rbf', 'poly', 'sigmoid'), 'C':[1, 10, 100, 1000], 'gamma': [0.001, 0.0001]},
              {'criterion':('gini', 'entropy'), 'max_features':[2, 3, 4, None], 'max_depth':[2,3,4,None]}]


def find_best_params(clf_list, param_list, features, labels):
    from sklearn.model_selection import GridSearchCV
    best_params=[]
    for clf, param in zip(clf_list, param_list):
        best_clf = GridSearchCV(clf(), param)
        best_clf.fit(features, labels)
        best_params.append( best_clf.best_params_ )
    return best_params

best_params = find_best_params(classifiers, parameters, features_train, labels_train)
print 'The best parameters for each classifiers:', best_params
print ''

# try different classifiers
clf_names = ['Gaussion Naive Bayes', 'SVM', 'Decision Tree']
results = {}
for name, clf_cls, param in zip(clf_names, classifiers, best_params):
    tmp_dict = {}
    # use KFold for cross validation, and input best parameters for each algorithm
    tmp_dict['precision'] = cross_val_score( clf_cls(**param), features, labels, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring='precision').mean()
    tmp_dict['recall'] = cross_val_score( clf_cls(**param), features, labels, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring='recall').mean()
    tmp_dict['F1 Score'] = cross_val_score( clf_cls(**param), features, labels, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring='f1').mean()
    results[name] = tmp_dict

# validation and metrics
pp.pprint(results)
print ''

### Task 6: Dump your classifier, dataset, and features_list.
# ['poi', 'bonus', 'deferred_income', 'expenses', 'total_payments', 'total_stock_value'] features which can achieve best performance

print "Final feature list:", features_list
clf = GaussianNB() # best classifier
print "Final classifier: Gaussian Naive Bayes"

'''
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
'''

dump_classifier_and_data(clf, my_dataset, features_list)
