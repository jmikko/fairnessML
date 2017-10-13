from load_data import load_binary_diabetes_uci, load_heart_uci, load_breast_cancer, load_adult, load_adult_race
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from measures import equalized_odds_measure_TP, equalized_odds_measure_FP, equalized_odds_measure_TP_from_list_of_sensfeat
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.optimize import linprog
from hardt import gamma_y_hat, HardtMethod
from scipy.spatial import ConvexHull
from collections import namedtuple


experiment_number = 0
if experiment_number == 0:
    dataset_train = load_binary_diabetes_uci()
    dataset_test = load_binary_diabetes_uci()
    sensible_feature = 1  # sex
elif experiment_number == 1:
    dataset_train = load_heart_uci()
    dataset_test = load_heart_uci()
    sensible_feature = 1  # sex
elif experiment_number == 2:
    dataset_train, dataset_test = load_adult(smaller=False)
    sensible_feature = 9  # sex
    print('Different values of the sensible feature', sensible_feature, ':',
          set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 3:
    dataset_train, dataset_test = load_adult_race(smaller=False)
    sensible_feature = 8  # race
    print('Different values of the sensible feature', sensible_feature, ':',
          set(dataset_train.data[:, sensible_feature]))

if experiment_number in [0, 1]:
    # % for train
    ntrain = 5 * len(dataset_train.target) // 10
    dataset_train.data = dataset_train.data[:ntrain, :]
    dataset_train.target = dataset_train.target[:ntrain]
    dataset_test.data = dataset_test.data[ntrain:, :]
    dataset_test.target = dataset_test.target[ntrain:]
if experiment_number in [2, 3]:
    ntrain = len(dataset_test.target)

# Standard SVM
# Train an SVM using the training set
print('Grid search...')
grid_search_complete = 1
if grid_search_complete:
    param_grid = [
        {'C': [0.1, 0.5, 1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [0.1, 0.5, 1, 10, 100, 1000], 'gamma': ['auto', 0.001, 0.0001], 'kernel': ['rbf']},
    ]
else:
    param_grid = [{'C': [10.0], 'kernel': ['linear'], 'gamma': ['auto']}]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid, n_jobs=1)
clf.fit(dataset_train.data, dataset_train.target)
print('Y:', clf.best_estimator_)

# Accuracy
pred = clf.predict(dataset_test.data)
pred_train = clf.predict(dataset_train.data)
print('Accuracy test:', accuracy_score(dataset_test.target, pred))
print('Accuracy train:', accuracy_score(dataset_train.target, pred_train))
# Fairness measure
print('Eq. opp. test: \n',
      equalized_odds_measure_TP(dataset_test, clf, [sensible_feature], ylabel=1))
print('Eq. opp. train: \n',
      equalized_odds_measure_TP(dataset_train, clf, [sensible_feature], ylabel=1))



# Our idea for fairness
values_of_sensible_feature = list(set(dataset_train.data[:, sensible_feature]))
list_of_sensible_feature_train = dataset_train.data[:, sensible_feature]
list_of_sensible_feature_test = dataset_test.data[:, sensible_feature]
val0 = np.min(values_of_sensible_feature)
val1 = np.max(values_of_sensible_feature)

tmp = [ex for idx, ex in enumerate(dataset_train.data)
       if dataset_train.target[idx] == 1 and ex[sensible_feature] == val1]
average_A_1 = np.mean(tmp, 0)
n_A_1 = len(tmp)

tmp = [ex for idx, ex in enumerate(dataset_train.data)
       if dataset_train.target[idx] == 1 and ex[sensible_feature] == val0]
average_not_A_1 = np.mean(tmp, 0)
n_not_A_1 = len(tmp)

N_1 = len([ex for idx, ex in enumerate(dataset_train.data) if dataset_train.target[idx] == 1])
#  print(len(average_A_1), len(average_not_A_1))
#  print(n_A_1, n_not_A_1, N_1)

u = (average_A_1 - average_not_A_1) * (n_A_1 * n_not_A_1) / N_1
#  print(u)
#  print(u[sensible_feature])

newdata = np.array([ex if ex[sensible_feature] == val0 else ex - u for ex in dataset_train.data])
#  newdata = map(lambda x: x - u, dataset_train.data)
newdata = np.delete(newdata, sensible_feature, 1)
#  print(newdata.shape)
dataset_train = namedtuple('_', 'data, target')(newdata, dataset_train.target)

newdata = np.array([ex if ex[sensible_feature] == val0 else ex - u for ex in dataset_test.data])
#  newdata = map(lambda x: x - u, dataset_test.data)
newdata = np.delete(newdata, sensible_feature, 1)
#  print(newdata.shape)
dataset_test = namedtuple('_', 'data, target')(newdata, dataset_test.target)

# Train an SVM using the training set
print('Grid search...')
grid_search_complete = 1
if grid_search_complete:
    param_grid = [
        {'C': [0.1, 0.5, 1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [0.1, 0.5, 1, 10, 100, 1000], 'gamma': ['auto', 0.001, 0.0001], 'kernel': ['rbf']},
    ]
else:
    param_grid = [{'C': [10.0], 'kernel': ['linear'], 'gamma': ['auto']}]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid, n_jobs=1)
clf.fit(dataset_train.data, dataset_train.target)
print('Y fair:', clf.best_estimator_)

# Accuracy
pred = clf.predict(dataset_test.data)
pred_train = clf.predict(dataset_train.data)
print('Accuracy test fair:', accuracy_score(dataset_test.target, pred))
print('Accuracy train fair:', accuracy_score(dataset_train.target, pred_train))
# Fairness measure
print('Eq. opp. test fair: \n',
      equalized_odds_measure_TP_from_list_of_sensfeat(dataset_test, clf, [list_of_sensible_feature_test], ylabel=1))
print('Eq. opp. train fair: \n',
      equalized_odds_measure_TP_from_list_of_sensfeat(dataset_train, clf, [list_of_sensible_feature_train], ylabel=1))
