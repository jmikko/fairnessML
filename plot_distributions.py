import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from load_data import load_binary_diabetes_uci, load_heart_uci, load_breast_cancer,\
    load_adult, load_adult_race, load_adult_race_white_vs_black, laod_propublica_fairml, laod_propublica_fairml_race,\
    laod_propublica_fairml_hotencoded, load_default, load_hepatitis, load_arrhythmia
from sklearn import svm
import numpy as np
from measures import equalized_odds_measure_TP, equalized_odds_measure_FP, equalized_odds_measure_from_pred_TP, equalized_odds_measure_TP_from_list_of_sensfeat
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.optimize import linprog
from hardt import gamma_y_hat, HardtMethod
from scipy.spatial import ConvexHull
from uncorrelation import UncorrelationMethod
from uncorrelation_nonlinear import Fair_SVM, polynomial_kernel, gaussian_kernel, linear_kernel
import os, sys
import numpy as np
from collections import namedtuple
#import plot_syn_boundaries as psb
from copy import deepcopy
import matplotlib.pyplot as plt  # for plotting stuff
from sklearn.metrics import recall_score
from sklearn.metrics.classification import _check_targets
from sklearn.metrics import make_scorer
from validation_method import two_step_validation_with_DEO

# Plot a 1D density example
experiment_number = 1

smaller_option = True
verbose = 3
iteration = 0
if experiment_number == 0:
    print('Loading diabetes dataset...')
    dataset_train = load_binary_diabetes_uci()
    dataset_test = load_binary_diabetes_uci()
    sensible_feature = 1  # sex
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 1:
    print('Loading heart dataset...')
    dataset_train = load_heart_uci()
    dataset_test = load_heart_uci()
    sensible_feature = 1  # sex
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 2:
    print('Loading adult (gender) dataset...')
    dataset_train, dataset_test = load_adult(smaller=smaller_option)
    sensible_feature = 9  # sex
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 3:
    print('Loading adult (white vs. other races) dataset...')
    dataset_train, dataset_test = load_adult_race(smaller=smaller_option)
    sensible_feature = 8  # race
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 4:
    print('Loading adult (gender) dataset by splitting the training data...')
    dataset_train, _ = load_adult(smaller=smaller_option)
    sensible_feature = 9  # sex
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 5:
    print('Loading adult (white vs. other races)  dataset by splitting the training data...')
    dataset_train, _ = load_adult_race(smaller=smaller_option)
    sensible_feature = 8  # race
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 6:
    print('Loading adult (white vs. black)  dataset by splitting the training data...')
    dataset_train, _ = load_adult_race_white_vs_black(smaller=smaller_option)
    sensible_feature = 8  # race
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 7:
    print('Loading propublica_fairml (gender) dataset with race not hotencoded...')
    dataset_train = laod_propublica_fairml()
    sensible_feature = 4  # gender
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 8:
    print('Loading propublica_fairml (black vs other races) dataset with race not hotencoded...')
    dataset_train = laod_propublica_fairml_race()
    sensible_feature = 5  # race
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 9:
    print('Loading propublica_fairml (gender) dataset with race hotencoded...')
    dataset_train = laod_propublica_fairml_hotencoded()
    sensible_feature = 10  # gender
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 10:
    print('Loading Default (gender) dataset [other categoricals are removed!]...')
    dataset_train = load_default(remove_categorical=True, smaller=smaller_option, scaler=True)
    sensible_feature = 1  # gender
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 11:
    print('Loading Hepatitis (gender) dataset...')
    dataset_train = load_hepatitis()
    sensible_feature = 2  # gender
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 12:
    print('Loading Arrhythmia (gender) dataset for task: Normal Vs All-the-others...')
    dataset_train = load_arrhythmia()
    sensible_feature = 1  # gender
    if verbose >= 1 and iteration == 0:
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))

if experiment_number in [0, 1]:
    # % for train
    ntrain = 5 * len(dataset_train.target) // 10
    ntest = len(dataset_train.target) - ntrain
    permutation = list(range(len(dataset_train.target)))
    np.random.shuffle(permutation)
    train_idx = permutation[:ntrain]
    test_idx = permutation[ntrain:]
    dataset_train.data = dataset_train.data[train_idx, :]
    dataset_train.target = dataset_train.target[train_idx]
    dataset_test.data = dataset_test.data[test_idx, :]
    dataset_test.target = dataset_test.target[test_idx]
if experiment_number in [2, 3]:
    ntrain = len(dataset_train.target)
    ntest = len(dataset_test.target)
if experiment_number in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
    # % for train
    ntrain = 9 * len(dataset_train.target) // 10
    ntest = len(dataset_train.target) - ntrain
    permutation = list(range(len(dataset_train.target)))
    np.random.shuffle(permutation)
    train_idx = permutation[:ntrain]
    test_idx = permutation[ntrain:]
    dataset_test = namedtuple('_', 'data, target')(dataset_train.data[test_idx, :], dataset_train.target[test_idx])
    dataset_train = namedtuple('_', 'data, target')(dataset_train.data[train_idx, :], dataset_train.target[train_idx])


swap_labels = False
if swap_labels:
    print('Labels => SWAPPED!')
    ypos = np.max(dataset_train.target)
    yneg = np.min(dataset_train.target)
    dataset_train.target = np.array([ypos if yy == yneg else yneg for yy in dataset_train.target])
    dataset_test.target = np.array([ypos if yy == yneg else yneg for yy in dataset_test.target])

grid_search_complete = True
n_jobs = 2

if grid_search_complete:
    if experiment_number in [10]:
        param_grid_linear = [
            {'C': [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'kernel': ['linear']}
        ]
        param_grid_all = [
            {'C': [0.1, 1.0, 2.0, 3.0, 4.0, 5.0], 'kernel': ['linear']},
            {'C': [0.1, 1.0, 2.0, 3.0, 4.0, 5.0], 'gamma': [0.1, 0.01], 'kernel': ['rbf']},
        ]
    else:
        param_grid_linear = [
            {'C': np.logspace(-4, 2, 15), 'kernel': ['linear']}
        ]
        param_grid_all = [
            {'C': np.logspace(-4, 2, 15), 'kernel': ['linear']},
            {'C': np.logspace(-4, 2, 15), 'gamma': [1.0, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
        ]
else:
    print('---> No grid search performed! <---')
    param_grid_linear = [{'C': [1.0], 'kernel': ['linear']}]
    param_grid_all = [{'C': [10.0], 'kernel': ['rbf'], 'gamma': [0.4]}]

if smaller_option:
    print('---> A smaller dataset could be loaded <---')

np.random.seed(1)
X = dataset_train.data
y = dataset_train.target
Xte = dataset_test.data
yte = dataset_test.target

ypos = np.max(y)
yneg = np.min(y)
sensible_feature_values = list(set([v[sensible_feature] for v in X]))
sensible_feature_values = [np.min(sensible_feature_values), np.max(sensible_feature_values)]


print('Linear SVM...')
# Train an SVM using the training set
print('\nGrid search for the standard Linear SVM...')
svc = svm.SVC()
score, best_estimator = two_step_validation_with_DEO(dataset_train, dataset_test, svc, verbose=verbose,
                                                     n_jobs=n_jobs,
                                                     sensible_feature=sensible_feature, params=param_grid_linear)
distance_from_hyperplane = best_estimator.decision_function(Xte)
idx_group_A1 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[0] and y[idx] == ypos]
idx_group_B1 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[1] and y[idx] == ypos]
idx_group_A0 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[0] and y[idx] == yneg]
idx_group_B0 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[1] and y[idx] == yneg]


print('Values:', distance_from_hyperplane)
xmin = np.min(distance_from_hyperplane)-0.2
xmax = np.max(distance_from_hyperplane)+0.2

print(best_estimator.predict(Xte))


bins = 30
fig, ax = plt.subplots(2, 1)
ax[0].hist(distance_from_hyperplane[idx_group_A1], bins=bins, normed=True, stacked=True, label='Group A1', alpha=1.0)
ax[1].hist(distance_from_hyperplane[idx_group_B1], bins=bins, normed=True, stacked=True, label='Group B1', alpha=1.0)
ax[0].hist(distance_from_hyperplane[idx_group_A0], bins=bins, normed=True, stacked=True, label='Group A0', alpha=0.5)
ax[1].hist(distance_from_hyperplane[idx_group_B0], bins=bins, normed=True, stacked=True, label='Group B0', alpha=0.5)
ax[0].axvline(x=best_estimator.intercept_, color='k')
ax[1].axvline(x=best_estimator.intercept_, color='k')
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
ax[0].set_xlim(left=xmin, right=xmax)
ax[1].set_xlim(left=xmin, right=xmax)
plt.title('Dataset #%d - SVM' % experiment_number)


fig, ax = plt.subplots()
pdf, bins, patches = ax.hist(distance_from_hyperplane[idx_group_A1], bins=bins, normed=True, stacked=True, label='Group A1', alpha=1.0)
# print(np.sum(pdf * np.diff(bins))) # it has to be 1!
ax.hist(distance_from_hyperplane[idx_group_B1], bins=bins, normed=True, stacked=True, label='Group B1', alpha=0.5)
ax.axvline(x=best_estimator.intercept_, color='k')
print('SVM intercept:', best_estimator.intercept_)
ax.legend(loc='upper left')
ax.set_xlim(left=xmin, right=xmax)
plt.title('Dataset #%d - SVM' % experiment_number)



print('Linear Fair SVM...')
# Train an SVM using the training set
print('\nGrid search for the Fair Linear SVM...')
svc = svm.SVC()
algorithm = UncorrelationMethod(dataset_train, model=None, sensible_feature=sensible_feature)
new_dataset_train = algorithm.new_representation(dataset_train.data)
new_dataset_train = namedtuple('_', 'data, target')(new_dataset_train, dataset_train.target)
new_dataset_test = algorithm.new_representation(dataset_test.data)
new_dataset_test = namedtuple('_', 'data, target')(new_dataset_test, dataset_test.target)
score, best_estimator = two_step_validation_with_DEO(new_dataset_train, new_dataset_test, svc, verbose=verbose,
                                                     n_jobs=n_jobs,
                                                     sensible_feature=sensible_feature, params=param_grid_linear,
                                                     list_of_sensible_feature=[x[sensible_feature] for x in
                                                                               dataset_train.data])
distance_from_hyperplane = best_estimator.decision_function(new_dataset_test.data)
idx_group_A1 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[0] and y[idx] == ypos]
idx_group_B1 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[1] and y[idx] == ypos]
idx_group_A0 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[0] and y[idx] == yneg]
idx_group_B0 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[1] and y[idx] == yneg]


print('Fair Values:', distance_from_hyperplane)
xmin = np.min(distance_from_hyperplane)-0.2
xmax = np.max(distance_from_hyperplane)+0.2


bins = 10
fig, ax = plt.subplots(2, 1)
ax[0].hist(distance_from_hyperplane[idx_group_A1], bins=bins, normed=True, stacked=True, label='Group A1', alpha=1.0)
ax[1].hist(distance_from_hyperplane[idx_group_B1], bins=bins, normed=True, stacked=True, label='Group B1', alpha=1.0)
ax[0].hist(distance_from_hyperplane[idx_group_A0], bins=bins, normed=True, stacked=True, label='Group A0', alpha=0.5)
ax[1].hist(distance_from_hyperplane[idx_group_B0], bins=bins, normed=True, stacked=True, label='Group B0', alpha=0.5)
ax[0].axvline(x=best_estimator.intercept_, color='k')
ax[1].axvline(x=best_estimator.intercept_, color='k')
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
ax[0].set_xlim(left=xmin, right=xmax)
ax[1].set_xlim(left=xmin, right=xmax)
plt.title('Dataset #%d - SVM FAIR' % experiment_number)


fig, ax = plt.subplots()
pdf, bins, patches = ax.hist(distance_from_hyperplane[idx_group_A1], bins=bins, normed=True, stacked=True, label='Group A1', alpha=1.0)
# print(np.sum(pdf * np.diff(bins))) # it has to be 1!
ax.hist(distance_from_hyperplane[idx_group_B1], bins=bins, normed=True, stacked=True, label='Group B1', alpha=0.5)
ax.axvline(x=best_estimator.intercept_, color='k')
print('Fair SVM intercept:', best_estimator.intercept_)
ax.legend(loc='upper left')
ax.set_xlim(left=xmin, right=xmax)
plt.title('Dataset #%d - SVM FAIR' % experiment_number)

plt.show()
