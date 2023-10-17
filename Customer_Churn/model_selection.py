
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN

df = pd.read_csv('/content/Tel__churn_eda.csv')
df.head(2)

df = df.drop('Unnamed: 0', axis = 1)
df.head()

x = df.drop('Churn', axis = 1)
x

y = df['Churn']
y

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

#Decision Tree Classifier

dt_model = DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth = 6, min_samples_leaf = 8)
dt_model.fit(x_train, y_train)

dt_y_pred = dt_model.predict(x_test)
dt_y_pred

dt_model.score(x_test, y_test)

print(classification_report(y_test, dt_y_pred))

sm = SMOTEENN()
x_sampled, y_sampled = sm.fit_resample(x,y)

xr_train, xr_test, yr_train, yr_test = train_test_split(x_sampled, y_sampled, test_size = 0.2)

dt_model_r = DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth = 6, min_samples_leaf = 8)

dt_model_r.fit(xr_train, yr_train)
dt_yr_pred = dt_model_r.predict(xr_test)
dt_r_score = dt_model_r.score(xr_test, yr_test)
print(f'Score: {dt_r_score} \n')
print(f'Report\n{classification_report(yr_test, dt_yr_pred)}')
print(f'Matrix: \n{confusion_matrix(yr_test, dt_yr_pred)}')

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 6, min_samples_leaf = 8, random_state = 100)

rf_model.fit(x_train, y_train)
rf_y_pred = rf_model.predict(x_test)

print(f'Score: {rf_model.score(x_test, y_test)}\n')
print(f'Report: \n{classification_report(y_test, rf_y_pred)}')
print(f'Matrix: \n{confusion_matrix(y_test, rf_y_pred)}')

rf_model_r = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 6, min_samples_leaf = 8, random_state = 100)

rf_model_r.fit(xr_train, yr_train)
rf_yr_pred = rf_model.predict(xr_test)

print(f'Score: {rf_model_r.score(xr_test, yr_test)}\n')
print(f'Report: \n{classification_report(yr_test, rf_yr_pred)}')
print(f'Matrix: \n{confusion_matrix(yr_test, rf_yr_pred)}')

#PCA

from sklearn.decomposition import PCA

pca = PCA(0.9)
xr_train_pca = pca.fit_transform(xr_train)
xr_test_pca = pca.transform(xr_test)
explained_variance = pca.explained_variance_ratio_

rf_model_pca = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_depth = 6, min_samples_leaf = 8, random_state = 100)
rf_model_pca.fit(xr_train_pca, yr_train)

yr_pca_pred = rf_model_pca.predict(xr_test_pca)

print(f'Score: {rf_model_pca.score(xr_test_pca, yr_test)}\n')
print(f'Report: \n{classification_report(yr_test, yr_pca_pred)}')
print(f'Matrix: \n{confusion_matrix(yr_test, yr_pca_pred)}')

#Pickling the Model

import pickle

filename = 'model.sav'
pickle.dump(rf_model_r, open(filename, 'wb'))

load_model = pickle.load(open(filename, 'rb'))

print(f'Score: {load_model.score(xr_test, yr_test)}')

