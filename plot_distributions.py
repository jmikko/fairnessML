import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from load_data import load_experiments
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
from toy_problem_lasso import toy_test_generator


SMALL_SIZE = 25
MEDIUM_SIZE = 25
BIGGER_SIZE = 28

bins = 15
# Plot a 1D density example
toytest = False

experiment_number = 2
smaller_option = True
verbose = 3
iteration = 0


if toytest:
    # Dataset
    n_samples = 100 * 10
    n_samples_low = 20 * 10

    lasso_dataset = False
    number_of_random_features = 100
    varA = 0.8
    aveApos = [-1.0, -1.0]
    aveAneg = [1.0, 1.0]
    varB = 0.5
    aveBpos = [0.5, -0.5]
    aveBneg = [0.5, 0.5]
    X, y, X_test, y_test, idx_A, idx_B, _, sensible_feature =\
        toy_test_generator(n_samples, n_samples_low, varA, aveApos, aveAneg, varB, aveBpos, aveBneg,
                           lasso_dataset, number_of_random_features)
    dataset_train = namedtuple('_', 'data, target')(X, y)
    dataset_test = namedtuple('_', 'data, target')(X_test, y_test)
else:
    dataset_train, dataset_test, sensible_feature = load_experiments(experiment_number,
                                                                        smaller_option,
                                                                        verbose)


swap_labels = False
if swap_labels:
    print('Labels => SWAPPED!')
    ypos = np.max(dataset_train.target)
    yneg = np.min(dataset_train.target)
    dataset_train.target = np.array([ypos if yy == yneg else yneg for yy in dataset_train.target])
    dataset_test.target = np.array([ypos if yy == yneg else yneg for yy in dataset_test.target])

grid_search_complete = True
n_jobs = 1

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
idx_group_A1 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[0] and yte[idx] == ypos]
idx_group_B1 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[1] and yte[idx] == ypos]
idx_group_A0 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[0] and yte[idx] == yneg]
idx_group_B0 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[1] and yte[idx] == yneg]


print('Values:', distance_from_hyperplane)
xmin = np.min(distance_from_hyperplane)
xmax = np.max(distance_from_hyperplane)

print('Predictions:', best_estimator.predict(Xte))


fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=80)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
ax[0].hist(distance_from_hyperplane[idx_group_A1], bins=bins, normed=True, stacked=True, label='A, Y=1', alpha=1.0)
ax[1].hist(distance_from_hyperplane[idx_group_B1], bins=bins, normed=True, stacked=True, label='B, Y=1', alpha=1.0)
ax[0].hist(distance_from_hyperplane[idx_group_A0], bins=bins, normed=True, stacked=True, label='A, Y=0', alpha=0.5)
ax[1].hist(distance_from_hyperplane[idx_group_B0], bins=bins, normed=True, stacked=True, label='B, Y=0', alpha=0.5)
# ax[0].axvline(x=best_estimator.intercept_, color='k')
# ax[1].axvline(x=best_estimator.intercept_, color='k')
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
ax[0].set_xlim(left=xmin, right=xmax)
ax[1].set_xlim(left=xmin, right=xmax)
plt.title('Dataset #%d - SVM' % experiment_number)


fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
pdf, bins, patches = ax.hist(distance_from_hyperplane[idx_group_A1], bins=bins, normed=True, stacked=True, label='A, Y=1', alpha=1.0)
# print(np.sum(pdf * np.diff(bins))) # it has to be 1!
ax.hist(distance_from_hyperplane[idx_group_B1], bins=bins, normed=True, stacked=True, label='B, Y=1', alpha=0.5)
# ax.axvline(x=best_estimator.intercept_, color='k')
# print('SVM intercept:', best_estimator.intercept_)
ax.legend(loc='upper left')
ax.set_xlim(left=xmin, right=xmax)
plt.title('Dataset #%d - SVM' % experiment_number)

if toytest:
    plt.title('Toytest - SVM')
    plt.savefig('d_toytest_svm')
else:
    plt.savefig('d_experiment-%d_svm' % experiment_number)



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
idx_group_A1 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[0] and yte[idx] == ypos]
idx_group_B1 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[1] and yte[idx] == ypos]
idx_group_A0 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[0] and yte[idx] == yneg]
idx_group_B0 = [idx for idx, v in enumerate(Xte) if v[sensible_feature] == sensible_feature_values[1] and yte[idx] == yneg]


print('Fair Values:', distance_from_hyperplane)
#xmin = np.min(distance_from_hyperplane)
#xmax = np.max(distance_from_hyperplane)
print('Predictions:', best_estimator.predict(new_dataset_test.data))


fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=80)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
ax[0].hist(distance_from_hyperplane[idx_group_A1], bins=bins, normed=True, stacked=True, label='A, Y=1', alpha=1.0)
ax[1].hist(distance_from_hyperplane[idx_group_B1], bins=bins, normed=True, stacked=True, label='B, Y=1', alpha=1.0)
ax[0].hist(distance_from_hyperplane[idx_group_A0], bins=bins, normed=True, stacked=True, label='A, Y=0', alpha=0.5)
ax[1].hist(distance_from_hyperplane[idx_group_B0], bins=bins, normed=True, stacked=True, label='B, Y=0', alpha=0.5)
# ax[0].axvline(x=best_estimator.intercept_, color='k')
# ax[1].axvline(x=best_estimator.intercept_, color='k')
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
ax[0].set_xlim(left=xmin, right=xmax)
ax[1].set_xlim(left=xmin, right=xmax)
plt.title('Dataset #%d - SVM FAIR' % experiment_number)


fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
pdf, bins, patches = ax.hist(distance_from_hyperplane[idx_group_A1], bins=bins, normed=True, stacked=True, label='A, Y=1', alpha=1.0)
# print(np.sum(pdf * np.diff(bins))) # it has to be 1!
ax.hist(distance_from_hyperplane[idx_group_B1], bins=bins, normed=True, stacked=True, label='B, Y=1', alpha=0.5)
# ax.axvline(x=best_estimator.intercept_, color='k')
# print('Fair SVM intercept:', best_estimator.intercept_)
ax.legend(loc='upper left')
ax.set_xlim(left=xmin, right=xmax)
plt.title('Dataset #%d - SVM FAIR' % experiment_number)

if toytest:
    plt.title('Toytest - Our Method')
    plt.savefig('d_toytest_fair')
else:
    plt.savefig('d_experiment-%d_fair' % experiment_number)


plt.show()
