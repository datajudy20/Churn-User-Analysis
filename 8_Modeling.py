"""
8단계
전체 모델링
** data_model.csv 저장 **
"""

import numpy as np
import pandas as pd
import time
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

from yellowbrick.classifier import ROCAUC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

start_time = time.time()

data = pd.read_csv("F:/통계논문/full_data.csv", encoding='utf-8',
                    dtype={'char': object,
                           'char_nth' : int,
                           'race_id': int,
                           'race': object,
                           'class_id': int,
                           'class': object,
                           'max_level': int,
                           'y' : int,
                           'date_length' : int,
                           'guild_ox': int,
                           'guild_change' : int,
                           'guild_level' : int,
                           'play_sec7_mean' : float,
                           'play_sec5_mean' : float,
                           'play_sec2_mean' : float,
                           'play_day7_mean' : float,
                           'play_day5_mean' : float,
                           'play_day2_mean' : float,
                           'orgrimmar' : int})

cols = data.columns[[1,19,2,3,4,5,10,17,18,11,12,13,14,15,16,20,7,9,8,6]]
data = data[cols]

#print(data.info())
#print(data.head())
#print(data.tail())
#print(data.describe())

for col in data.columns:
       print(col)
       print(np.unique(data[col]), end='\n\n')

drop_char = []

sec5_na = data.loc[data['play_sec5_mean'].isna(), 'char']
for char in sec5_na:
       tmp = data.loc[data['char'] == char,
                      ['play_sec7_mean', 'play_sec2_mean',
                       'date_length', 'start_level', 'max_level']]
       if tmp['play_sec7_mean'].values == tmp['play_sec2_mean'].values:
           data.loc[data['char'] == char, ['play_sec5_mean']] = 0
       else:
              drop_char.append(char)
              #print(char)
              #print(tmp)


sec2_na = data.loc[data['play_sec2_mean'].isna(), 'char']
for char in sec2_na:
       tmp = data.loc[data['char'] == char,
                      ['play_sec7_mean', 'play_sec5_mean',
                       'date_length', 'start_level', 'max_level']]
       if tmp['play_sec7_mean'].values == tmp['play_sec5_mean'].values:
              data.loc[data['char'] == char, ['play_sec2_mean']] = 0
       else:
              drop_char.append(char)
              #print(char)
              #print(tmp)

print(data[data['play_sec7_mean']>86000].index)
print(data[data['play_sec5_mean']>86000].index)
print(data[data['play_sec2_mean']>86000].index)
data=data.drop(data[data['play_sec7_mean']>86000].index)
data=data.drop(data[data['play_sec5_mean']>86000].index)
data=data.drop(data[data['play_sec2_mean']>86000].index)


#print(len(drop_char))
print(data['y'].value_counts())
data.drop(data[data['char'].isin(drop_char)].index, inplace=True)
print(data['y'].value_counts())
print(data.info())

for col in data.columns:
    print(col)
    print('unique 값 개수 : ', len(np.unique(data[col])))
    print('최소값 : ', np.min(np.unique(data[col])))
    print('최대값 : ', np.max(np.unique(data[col])))
    print()


## 모델용 데이터 저장
data.to_csv("F:/통계논문/data_model.csv", mode='w', index=False)


data = pd.read_csv("F:/통계논문/data_model.csv", encoding='utf-8',
                    dtype={'char': object,
                           'char_nth' : int,
                           'race_id': int,
                           'race': object,
                           'class_id': int,
                           'class': object,
                           'max_level': int,
                           'y' : int,
                           'date_length' : int,
                           'guild_ox': int,
                           'guild_change' : int,
                           'guild_level' : int,
                           'play_sec7_mean' : float,
                           'play_sec5_mean' : float,
                           'play_sec2_mean' : float,
                           'play_day7_mean' : float,
                           'play_day5_mean' : float,
                           'play_day2_mean' : float,
                           'orgrimmar' : int})


print(data['y'].value_counts())

print(data.info())
print(data.head())
print(data.tail())
print(data.describe())

data_model = data.loc[:, ['char', 'race', 'class', 'guild_ox', 'y']]
# print(data_model.isnull().sum())

data_model['race'] = data_model['race'].astype(str)
data_model['class'] = data_model['class'].astype(str)
data_model_dum = pd.get_dummies(data_model.drop(columns=['char']), drop_first=True)

# print(data_model_dum.info())
# print(data_model_dum.head())
# print(data_model_dum.tail())


### train / test 분리 ###
X = data_model_dum.drop(columns=['y']).copy()
y = data_model_dum['y']
# print(X.columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values,
                                                    shuffle=True, test_size=0.25,
                                                    random_state=0)
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# print("----- train shape -----")
# print(X_train.shape, y_train.shape)
# print(pd.DataFrame(y_train).value_counts())
# print("----- test shape -----")
# print(X_test.shape, y_test.shape)
# print(pd.DataFrame(y_test).value_counts())


### 데이터 표준화 ###
from sklearn.preprocessing import StandardScaler

scaler_std = StandardScaler()
X_train = scaler_std.fit_transform(X_train)
X_test = scaler_std.transform(X_test)


### KNN ###
for i in range(3, 12, 2):
    knn = KNeighborsClassifier(n_neighbors=i, metric='mahalanobis', metric_params={'V': np.cov(X_train.T)})
    knn.fit(X_train, y_train)

    print("----- k=" + str(i) + ": train score -----")
    print(knn.score(X_train, y_train))
    y_pred = knn.predict(X_train)
    print('f1 score : ', metrics.f1_score(y_train, y_pred))

    print("----- k=" + str(i) + ":  test score -----")
    print(knn.score(X_test, y_test))
    y_pred = knn.predict(X_test)
    print('f1 score : ', metrics.f1_score(y_test, y_pred))

model_knn = KNeighborsClassifier(n_neighbors=7, metric='mahalanobis', metric_params={'V': np.cov(X_train.T)})
model_knn.fit(X_train, y_train)
y_pred = model_knn.predict(X_test)
print("----- test score -----")
print('accuracy', metrics.accuracy_score(y_test, y_pred))
print('precision : ', metrics.precision_score(y_test, y_pred))

print("----- confusion matrix -----")
print(confusion_matrix(y_test, y_pred))

print('--- ROC curve ---')
y_proba = model_knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_proba)
auc = metrics.roc_auc_score(y_test, y_proba)
print('auc score : ', auc)
fig = plt.figure()
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC Curve - KNN')
fig.savefig('F:/통계논문/image/roc_knn.png')

perm = permutation_importance(model_knn, X_train, y_train, n_repeats=10,
                              random_state=0, n_jobs=-1)
sorted_idx = perm.importances_mean.argsort()
fig = plt.figure()
fig, ax = plt.subplots()
ax.boxplot(perm.importances[sorted_idx].T,
           vert=False, labels=X.columns[sorted_idx])
fig.tight_layout()
fig.savefig('F:/통계논문/image/permutation_knn_full.png')


### SVM ###
print('------- svm -------')

model_svm = SVC()
param_grid = {'C': [0.01, 0.1, 1, 10, 100],
              'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf'],
              'probability':[True]}

cv_svm = GridSearchCV(estimator=model_svm, param_grid=param_grid,
                  scoring='accuracy', cv=5)
cv_svm = cv_svm.fit(X_train, y_train)
best_params = cv_svm.best_params_
best_score = cv_svm.best_score_
best_svm = cv_svm.best_estimator_
print('best score: ', best_score)
print('best parameter : ', best_params)

#model_svm = SVC(**best_params)
model_svm = SVC(C=10, gamma=0.001, kernel='rbf', probability=True)
model_svm.fit(X_train, y_train)

print("----- train score -----")
print('accuracy : ', model_svm.score(X_train, y_train))

print("----- test score -----")
y_pred = model_svm.predict(X_test)
base_score = model_svm.score(X_test, y_test)
print('accuracy : ', base_score)
print('precision : ', metrics.precision_score(y_test, y_pred))

print('--- confusion matrix ---')
print(confusion_matrix(y_test, y_pred))

print('--- ROC curve ---')
y_proba = model_svm.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_proba)
auc = metrics.roc_auc_score(y_test, y_proba)
print('auc score : ', auc)
fig = plt.figure()
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC Curve - SVM')
fig.savefig('F:/통계논문/image/roc_svm.png')


perm = permutation_importance(model_svm, X_train, y_train, n_repeats=10,
                              random_state=0, n_jobs=-1)
sorted_idx = perm.importances_mean.argsort()
fig = plt.figure()
fig, ax = plt.subplots()
ax.boxplot(perm.importances[sorted_idx].T,
           vert=False, labels=X.columns[sorted_idx])
fig.tight_layout()
fig.savefig('F:/통계논문/image/permutation_svm_full.png')



### random forest ###
print('------- random forest -------')

model_rf = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 80, 100],
    'max_depth': [2, 3, 4],
    'criterion': ['gini'],
    'random_state': [0]}

cv_rf = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5)
cv_rf.fit(X_train, y_train)
best_params = cv_rf.best_params_
best_score = cv_rf.best_score_
best_rf = cv_rf.best_estimator_
print('best score: ', best_score)
print('best parameter : ', best_params)

#model_rf = RandomForestClassifier(**best_params)
model_rf = RandomForestClassifier(n_estimators=100, max_depth=3,
                                  criterion='gini', random_state=0)
model_rf.fit(X_train, y_train)

print('--- train score ---')
print('accuracy :', model_rf.score(X_train, y_train))

print('--- test score ---')
y_pred = model_rf.predict(X_test)
base_score = model_rf.score(X_test, y_test)
print('accuracy :', base_score)
print('precision : ', metrics.precision_score(y_test, y_pred))

print('--- confusion matrix ---')
print(confusion_matrix(y_test, y_pred))

print('--- ROC curve ---')
y_proba = model_rf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_proba)
auc = metrics.roc_auc_score(y_test, y_proba)
print('auc score : ', auc)
fig = plt.figure()
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC Curve - Random Forest')
fig.savefig('F:/통계논문/image/roc_rf.png')


importance = pd.DataFrame({'feature' : X.columns,
                           'importance' : model_rf.feature_importances_}).sort_values(by='importance', ascending=True)

fig = plt.figure()
plt.barh(importance['feature'], importance['importance'])
fig.tight_layout()
fig.savefig('F:/통계논문/image/importance_rf_full.png')


perm = permutation_importance(model_rf, X_train, y_train, n_repeats=10,
                              random_state=0, n_jobs=-1)
sorted_idx = perm.importances_mean.argsort()

fig = plt.figure()
fig, ax = plt.subplots()
ax.boxplot(perm.importances[sorted_idx].T,
           vert=False, labels=X.columns[sorted_idx])
fig.tight_layout()
fig.savefig('F:/통계논문/image/permutation_rf_full.png')


### XGBoost ###
print('------- xgboost -------')
import xgboost as xgb

model_xgb = xgb.XGBClassifier()
param_grid = {
    'use_label_encoder': [False],
    'n_estimators': [300, 400, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.05, 0.1],
    'min_child_weight': [2,3,4],
    'subsample': [0.7, 0.8],
    'eval_metric': ['error'],
    'n_jobs': [-1],
    'random_state': [0]}

cv_xgb = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5)
cv_xgb.fit(X_train, y_train)
best_params = cv_xgb.best_params_
best_score = cv_xgb.best_score_
best_xgb = cv_xgb.best_estimator_
print('best score: ', best_score)
print('best parameter : ', best_params)

#model_xgb = xgb.XGBClassifier(**best_params)
model_xgb = xgb.XGBClassifier(eval_metric='error',learning_rate= 0.05,
                              max_depth= 4, min_child_weight= 2,
                              n_estimators= 400, n_jobs= -1,
                              random_state= 0, subsample= 0.8,
                              use_label_encoder= False)
model_xgb.fit(X_train, y_train, early_stopping_rounds=10,
              eval_set=[(X_test, y_test)])


print('--- train score ---')
print('accuracy :', model_xgb.score(X_train, y_train))

print('--- test score ---')
y_pred = model_xgb.predict(X_test)
base_score = model_xgb.score(X_test, y_test)
print('accuracy :', base_score)
print('precision : ', metrics.precision_score(y_test, y_pred))

print('--- confusion matrix ---')
print(confusion_matrix(y_test, y_pred))

print('--- ROC curve ---')
y_proba = model_xgb.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_proba)
auc = metrics.roc_auc_score(y_test, y_proba)
print('auc score : ', auc)
fig = plt.figure()
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC Curve - XGBoost')
fig.savefig('F:/통계논문/image/roc_xgb.png')


importance = pd.DataFrame({'feature' : X.columns,
                           'importance' : model_xgb.feature_importances_}).sort_values(by='importance', ascending=True)

fig = plt.figure()
plt.barh(importance['feature'], importance['importance'])
fig.tight_layout()
fig.savefig('F:/통계논문/image/importance_xgb_full.png')


perm = permutation_importance(model_xgb, X_train, y_train, n_repeats=10,
                              random_state=0, n_jobs=-1)
sorted_idx = perm.importances_mean.argsort()
fig = plt.figure()
fig, ax = plt.subplots()
ax.boxplot(perm.importances[sorted_idx].T,
           vert=False, labels=X.columns[sorted_idx])
fig.tight_layout()
fig.savefig('F:/통계논문/image/permutation_xgb_full.png')


### Ensemble ###
print('------- ensemble -------')
from sklearn.ensemble import VotingClassifier

svm = SVC(probability=True, kernel='rbf')
rf = RandomForestClassifier(criterion='gini', random_state=0)
xgb = xgb.XGBClassifier(eval_metric='error',  n_jobs= -1,
                              random_state= 0, use_label_encoder= False)
knn = KNeighborsClassifier(metric='mahalanobis', metric_params={'V': np.cov(X_train.T)})
ensemble = VotingClassifier(estimators=[('svm',svm),
                                        ('rf', rf),
                                        ('xgb', xgb),
                                        ('knn', knn)],
                            voting='soft', n_jobs=-1)

param_svm_c = [1, 10]
param_svm_gamma = [0.001, 0.01]
param_rf_n = [100]
param_rf_depth = [3, 4]
param_xgb_n = [300, 400]
param_xgb_depth = [4, 5]
param_xgb_rate = [0.05]
param_xgb_child = [2]
param_xgb_samp = [0.8]
param_knn_n = [5, 7]
param_grid = [{'svm__C':param_svm_c, 'svm__gamma':param_svm_gamma,
               'rf__n_estimators':param_rf_n, 'rf__max_depth':param_rf_depth,
               'xgb__learning_rate':param_xgb_rate, 'xgb__n_estimators':param_xgb_n,
               'xgb__max_depth':param_xgb_depth, 'xgb__subsample':param_xgb_samp,
               'xgb__min_child_weight':param_xgb_child, 'knn__n_neighbors':param_knn_n}]

ensemble_cv = GridSearchCV(estimator=ensemble, param_grid=param_grid,
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
ensemble_cv = ensemble_cv.fit(X_train, y_train)
print('Best accuracy score: %.3f \nBest parameters: %s' % (ensemble_cv.best_score_, ensemble_cv.best_params_))

model_ensemble = ensemble_cv.best_estimator_
model_ensemble.fit(X_train, y_train)



ensemble = VotingClassifier(estimators=[('svm',model_svm),
                                        ('rf', model_rf),
                                        ('xgb', model_xgb),
                                        ('knn', model_knn)],
                            voting='soft', n_jobs=-1)

model_ensemble = ensemble.fit(X_train, y_train)

print('--- train score ---')
print('accuracy :', model_ensemble.score(X_train, y_train))

print('--- test score ---')
y_pred = model_ensemble.predict(X_test)
base_score = model_ensemble.score(X_test, y_test)
print('accuracy :', base_score)
print('precision : ', metrics.precision_score(y_test, y_pred))

print('--- confusion matrix ---')
print(confusion_matrix(y_test, y_pred))

print('--- ROC curve ---')
y_proba = model_ensemble.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_proba)
auc = metrics.roc_auc_score(y_test, y_proba)
print('auc score : ', auc)


fig = plt.figure()
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC Curve - Ensemble')
fig.savefig('F:/통계논문/image/roc_ensemble.png')

perm = permutation_importance(model_ensemble, X_train, y_train, n_repeats=10,
                              random_state=0, n_jobs=-1)
sorted_idx = perm.importances_mean.argsort()
fig = plt.figure()
fig, ax = plt.subplots()
ax.boxplot(perm.importances[sorted_idx].T,
           vert=False, labels=X.columns[sorted_idx])
fig.tight_layout()
fig.savefig('F:/통계논문/image/permutation_ensemble_full.png')


spent_time = time.time() - start_time
mins_spent = int(spent_time / 60)
secs_remainder = int(spent_time % 60)
print('Time of process: ', mins_spent, ':', secs_remainder)