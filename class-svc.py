# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:43:35 2019

@author: Hamid.t
"""

from sklearn import svm
#import numpy as np
#import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import NuSVC

from sklearn.metrics import accuracy_score
# Load wine dataset
cancer = load_breast_cancer()
columns_names = cancer.feature_names

y = cancer.target
X = cancer.data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)  

y_pred=clf.predict(X_test)

print('acuuracy  of svc of S is: ',accuracy_score(y_test, y_pred))
print('f1_score of svc is: ',f1_score(y_test,y_pred))



lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)

y_pred_lin=lin_clf.predict(X_test) 
print('acuuracy of linear svc is: ',accuracy_score(y_test, y_pred_lin))
print('f1_score of linear svc is: ',f1_score(y_test,y_pred_lin))

nu_clf = NuSVC(gamma='scale')
nu_clf.fit(X_train, y_train)

y_pred_nu=clf.predict(X_test)

print('acuuracy of Nu_SVC is: ',accuracy_score(y_test, y_pred_nu))
print('f1_score of NU_SVC is: ',f1_score(y_test,y_pred_nu))
 