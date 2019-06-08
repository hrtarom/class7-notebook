# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:15:21 2019

@author: Hamid.t
"""
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.datasets import load_breast_cancer
import os
cancer = load_breast_cancer()

#df  = pd.DataFrame(data.data)
#
#df.columns= data.feature_names
#df['target_label']=data.target

os.makedirs('./plots/comparative scatter', exist_ok=True)


y = cancer.target
X = cancer.data

# Splitting features and target datasets into: train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Training a Linear Regression model with fit()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lr.predict(X_test)

# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

# Printing accuracy score(mean accuracy) from 0 - 1
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# Printing the classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print('Classification Report')
print(classification_report(y_test, predicted_values))

# Printing the classification confusion matrix (diagonal is true)
print('Confusion Matrix')
print(confusion_matrix(y_test, predicted_values))

print('Overall f1-score')
print(f1_score(y_test, predicted_values, average="macro"))