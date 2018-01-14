from load_data import load_binary_diabetes_uci, load_heart_uci, load_breast_cancer,\
    load_adult, load_adult_race, load_adult_race_white_vs_black, laod_propublica_fairml, laod_propublica_fairml_race,\
    laod_propublica_fairml_hotencoded, load_default, load_hepatitis, load_arrhythmia
from load_data import load_experiments
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from measures import fair_tpr_from_precomputed, subgrups_sensible_feature_data
from uncorrelation import UncorrelationMethod
from collections import namedtuple
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from validation_method import two_step_validation_with_DEO
from collections import namedtuple
from toy_problem_lasso import toy_test_generator
from sklearn.linear_model import Lasso


class LassoC(Lasso):
    def predict(self, X):
        return np.sign(np.sign(super().predict(X)) + 0.1)

np.random.seed(15)
param_grid_linear = {'C': np.logspace(-6, 6, 40)}
#param_grid_linear['C'] = param_grid_linear['C'][10:-12]


toytest = True
lasso_algorithm = False
evaluate_approx_on_train = False

if toytest:
    # Dataset
    n_samples = 100 * 10
    n_samples_low = 20 * 10
    lasso_dataset = False
    number_of_random_features = 2000
    varA = 0.8
    aveApos = [-1.0, -1.0]
    aveAneg = [1.0, 1.0]
    varB = 0.5
    aveBpos = [0.5, -0.5]
    aveBneg = [0.5, 0.5]
    X, y, X_test, y_test, idx_A, idx_B, _, sensible_feature_id =\
        toy_test_generator(n_samples, n_samples_low, varA, aveApos, aveAneg, varB, aveBpos, aveBneg,
                           lasso_dataset, number_of_random_features)
    dataset_train = namedtuple('_', 'data, target')(X, y)
    dataset_test = namedtuple('_', 'data, target')(X_test, y_test)
else:
    experiment_number = 14
    iteration = 0
    verbose = 3
    smaller_option = True
    dataset_train, dataset_test, sensible_feature_id = load_experiments(experiment_number,
                                                                        smaller_option,
                                                                        verbose)

not_fair_stats = {'error': [], 'deo': [], 'EO_prod': [], 'deo_approx': []}
fair_stats = {'error': [], 'deo': [], 'EO_prod': [], 'deo_approx': [], 'delta0': [], 'delta1': []}

if not lasso_algorithm:
    # Not fair err\deo values:
    for C in param_grid_linear['C']:
        estimator = svm.LinearSVC(C=C, fit_intercept=True)
        estimator.fit(dataset_train.data, dataset_train.target)
        prediction = estimator.predict(dataset_test.data)
        error = 1.0 - accuracy_score(dataset_test.target, prediction)
        subgropus_idxs = subgrups_sensible_feature_data(dataset_test.data, sensible_feature_id)
        deo = fair_tpr_from_precomputed(dataset_test.target, prediction, subgropus_idxs)
        val0 = np.min(list(deo.keys()))
        val1 = np.max(list(deo.keys()))
        not_fair_stats['error'].append(error)
        not_fair_stats['deo'].append(np.abs(deo[val0] - deo[val1]))

        if evaluate_approx_on_train:
            adeo0 = np.mean([estimator.decision_function([ex]) for idx, ex in enumerate(dataset_train.data)
                             if dataset_train.target[idx] == 1 and dataset_train.data[idx][sensible_feature_id] == val0])
            adeo1 = np.mean([estimator.decision_function([ex]) for idx, ex in enumerate(dataset_train.data)
                             if dataset_train.target[idx] == 1 and dataset_train.data[idx][sensible_feature_id] == val1])
        else:
            adeo0 = np.mean([estimator.decision_function([ex]) for idx, ex in enumerate(dataset_test.data)
                             if dataset_test.target[idx] == 1 and dataset_test.data[idx][sensible_feature_id] == val0])
            adeo1 = np.mean([estimator.decision_function([ex]) for idx, ex in enumerate(dataset_test.data)
                             if dataset_test.target[idx] == 1 and dataset_test.data[idx][sensible_feature_id] == val1])
        not_fair_stats['deo_approx'].append(np.abs(adeo0 - adeo1))
        #  not_fair_stats['EO_prod'].append(deo[val0] * deo[val1])
        print('Coeff SVM near zero (C=', C, ') :', len([coef for coef in estimator.coef_[0] if coef < 1e-8]),
              '- error:', error, '- EO:', deo, ' DEO:', np.abs(deo[val0] - deo[val1]), 'AppDEO:', np.abs(adeo0 - adeo1))

    # Fair err\deo values:
    for C in param_grid_linear['C']:
        estimator = svm.LinearSVC(C=C, fit_intercept=True)
        algorithm = UncorrelationMethod(dataset_train, model=None, sensible_feature=sensible_feature_id)
        new_dataset_train = algorithm.new_representation(dataset_train.data)
        new_dataset_train = namedtuple('_', 'data, target')(new_dataset_train, dataset_train.target)
        new_dataset_test = algorithm.new_representation(dataset_test.data)
        new_dataset_test = namedtuple('_', 'data, target')(new_dataset_test, dataset_test.target)
        estimator.fit(new_dataset_train.data, new_dataset_train.target)
        prediction = estimator.predict(new_dataset_test.data)
        error = 1.0 - accuracy_score(dataset_test.target, prediction)
        subgropus_idxs = subgrups_sensible_feature_data(dataset_test.data, sensible_feature_id)
        deo = fair_tpr_from_precomputed(dataset_test.target, prediction, subgropus_idxs)
        val0 = np.min(list(deo.keys()))
        val1 = np.max(list(deo.keys()))
        fair_stats['error'].append(error)
        fair_stats['deo'].append(np.abs(deo[val0] - deo[val1]))

        if evaluate_approx_on_train:
            adeo0 = np.mean([estimator.decision_function([ex]) for idx, ex in enumerate(new_dataset_train.data)
                             if dataset_train.target[idx] == 1 and dataset_train.data[idx][sensible_feature_id] == val0])
            adeo1 = np.mean([estimator.decision_function([ex]) for idx, ex in enumerate(new_dataset_train.data)
                             if dataset_train.target[idx] == 1 and dataset_train.data[idx][sensible_feature_id] == val1])
        else:
            adeo0 = np.mean([estimator.decision_function([ex]) for idx, ex in enumerate(new_dataset_test.data)
                             if dataset_test.target[idx] == 1 and dataset_test.data[idx][sensible_feature_id] == val0])
            adeo1 = np.mean([estimator.decision_function([ex]) for idx, ex in enumerate(new_dataset_test.data)
                             if dataset_test.target[idx] == 1 and dataset_test.data[idx][sensible_feature_id] == val1])
        delta0 = np.abs(deo[val0] - 0.5 - adeo0)
        delta1 = np.abs(deo[val1] - 0.5 - adeo1)
        fair_stats['deo_approx'].append(np.abs(adeo0 - adeo1))
        fair_stats['delta0'].append(delta0)
        fair_stats['delta1'].append(delta1)
        #  fair_stats['EO_prod'].append(deo[val0] * deo[val1])
        print('Coeff Fair-SVM near zero (C=', C, ') :', len([coef for coef in estimator.coef_[0] if coef < 1e-8]),
              '- error:', error, '- EO:', deo, ' DEO:', np.abs(deo[val0] - deo[val1]), 'AppDEO:', np.abs(adeo0 - adeo1),
              '\nDelta0:', delta0, 'Delta1:', delta1)

else: #LASSO ALGO
    # Not fair err\deo values:
    for alpha in param_grid_linear['C']:
        estimator = LassoC(alpha=alpha, fit_intercept=True)
        estimator.fit(dataset_train.data, dataset_train.target)
        prediction = estimator.predict(dataset_test.data)
        error = 1.0 - accuracy_score(dataset_test.target, prediction)
        subgropus_idxs = subgrups_sensible_feature_data(dataset_test.data, sensible_feature_id)
        deo = fair_tpr_from_precomputed(dataset_test.target, prediction, subgropus_idxs)
        val0 = np.min(list(deo.keys()))
        val1 = np.max(list(deo.keys()))
        print('Coeff Lasso near zero (alpha=', alpha, ') :', len([coef for coef in estimator.coef_ if coef < 1e-8]),
              '- error:', error, '- EO:', deo, ' DEO:', np.abs(deo[val0] - deo[val1]))
        not_fair_stats['error'].append(error)
        not_fair_stats['deo'].append(np.abs(deo[val0] - deo[val1]))
        #  not_fair_stats['EO_prod'].append(deo[val0] * deo[val1])

    # Fair err\deo values:
    for alpha in param_grid_linear['C']:
        estimator = LassoC(alpha=alpha, fit_intercept=True)
        algorithm = UncorrelationMethod(dataset_train, model=None, sensible_feature=sensible_feature_id)
        new_dataset_train = algorithm.new_representation(dataset_train.data)
        new_dataset_train = namedtuple('_', 'data, target')(new_dataset_train, dataset_train.target)
        new_dataset_test = algorithm.new_representation(dataset_test.data)
        new_dataset_test = namedtuple('_', 'data, target')(new_dataset_test, dataset_test.target)
        estimator.fit(new_dataset_train.data, new_dataset_train.target)
        prediction = estimator.predict(new_dataset_test.data)
        error = 1.0 - accuracy_score(dataset_test.target, prediction)
        subgropus_idxs = subgrups_sensible_feature_data(dataset_test.data, sensible_feature_id)
        deo = fair_tpr_from_precomputed(dataset_test.target, prediction, subgropus_idxs)
        val0 = np.min(list(deo.keys()))
        val1 = np.max(list(deo.keys()))
        print('Coeff Fair-Lasso near zero (alpha=', alpha, ') :', len([coef for coef in estimator.coef_ if coef < 1e-8]),
              '- error:', error, '- EO:', deo, ' DEO:', np.abs(deo[val0] - deo[val1]))
        fair_stats['error'].append(error)
        fair_stats['deo'].append(np.abs(deo[val0] - deo[val1]))
        #  fair_stats['EO_prod'].append(deo[val0] * deo[val1])

print('Not-fair STATS:', not_fair_stats)
print('Not-fair smallest error:', np.min(not_fair_stats['error']))
print('Not-fair smallest deo:', np.min(not_fair_stats['deo']))
print('Fair STATS:', fair_stats)
print('Fair smallest error:', np.min(fair_stats['error']))
print('Fair smallest deo:', np.min(fair_stats['deo']))

besterr = np.array(fair_stats['error']).argsort()[:5]
print('Best with err:', np.array(fair_stats['error'])[besterr])
print('Best with deo:', np.array(fair_stats['deo'])[besterr])
bestdelta0 = np.array(fair_stats['delta0'])[besterr]
bestdelta1 = np.array(fair_stats['delta1'])[besterr]
print('Delta0 (over the best 5 errors):', np.mean(bestdelta0), '+-', np.std(bestdelta0))
print('Delta1 (over the best 5 errors):', np.mean(bestdelta1), '+-', np.std(bestdelta1))

SMALL_SIZE = 25
MEDIUM_SIZE = 25
BIGGER_SIZE = 28
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

fig = plt.figure(1, figsize=(9, 8), dpi=80)
ax = plt.subplot(111)
plt.plot(fair_stats['error'], fair_stats['deo'], 'o', markersize=15, label='Our method')
plt.plot(not_fair_stats['error'], not_fair_stats['deo'], '*', markersize=15, label='SVM')
plt.xlabel('Error')
plt.ylabel('DEO')
plt.legend()
if toytest:
    if not lasso_dataset:
        strtitle = 'Toytest'
    else:
        strtitle = 'Lasso_Toytest'
    if lasso_algorithm:
        strtitle += '-Lasso_Algorithms'
    plt.title(strtitle)
    plt.savefig(strtitle)
else:
    strtitle = 'Experiment_%d' % experiment_number
    if lasso_algorithm:
        strtitle += '-Lasso_ Algorithms'
    plt.title(strtitle)
    plt.savefig(strtitle)

if not lasso_algorithm:
    fig = plt.figure(2, figsize=(9, 8), dpi=80)
    ax = plt.subplot(111)
    plt.semilogx(param_grid_linear['C'], fair_stats['deo_approx'], '-o', markersize=5, label='Approx DEO')
    plt.semilogx(param_grid_linear['C'], fair_stats['deo'], '--x', markersize=10, label='DEO')
    plt.xlabel('C')
    # plt.ylabel('Value')
    plt.legend()
    if toytest:
        if not lasso_dataset:
            strtitle = 'Toytest - Our method'
        else:
            strtitle = 'Lasso_Toytest - Our method'
        plt.title(strtitle)
        plt.savefig(strtitle)
    else:
        strtitle = 'Experiment_%d - Our method' % experiment_number
        plt.title(strtitle)
        plt.savefig(strtitle)

    fig = plt.figure(3, figsize=(9, 8), dpi=80)
    ax = plt.subplot(111)
    plt.semilogx(param_grid_linear['C'], fair_stats['error'], '-D', markersize=5, label='Error')
    plt.xlabel('C')
    # plt.ylabel('Value')
    plt.legend()
    if toytest:
        if not lasso_dataset:
            strtitle = 'Toytest Error - Our method'
        else:
            strtitle = 'Lasso_Toytest Error - Our method'
        plt.title(strtitle)
        plt.savefig(strtitle)
    else:
        strtitle = 'Experiment_%d Error - Our method' % experiment_number
        plt.title(strtitle)
        plt.savefig(strtitle)

    fig = plt.figure(4, figsize=(9, 8), dpi=80)
    ax = plt.subplot(111)
    plt.semilogx(param_grid_linear['C'], not_fair_stats['deo_approx'], '-o', markersize=5, label='Approx DEO')
    plt.semilogx(param_grid_linear['C'], not_fair_stats['deo'], '--x', markersize=10, label='DEO')
    plt.xlabel('C')
    # plt.ylabel('Value')
    plt.legend()
    if toytest:
        if not lasso_dataset:
            strtitle = 'Toytest - SVM'
        else:
            strtitle = 'Lasso_Toytest - SVM'
        plt.title(strtitle)
        plt.savefig(strtitle)
    else:
        strtitle = 'Experiment_%d - SVM' % experiment_number
        plt.title(strtitle)
        plt.savefig(strtitle)

    fig = plt.figure(5, figsize=(9, 8), dpi=80)
    ax = plt.subplot(111)
    plt.semilogx(param_grid_linear['C'], not_fair_stats['error'], '-D', markersize=5, label='Error')
    plt.xlabel('C')
    # plt.ylabel('Value')
    plt.legend()
    if toytest:
        if not lasso_dataset:
            strtitle = 'Toytest Error - SVM'
        else:
            strtitle = 'Lasso_Toytest Error - SVM'
        plt.title(strtitle)
        plt.savefig(strtitle)
    else:
        strtitle = 'Experiment_%d Error - SVM' % experiment_number
        plt.title(strtitle)
        plt.savefig(strtitle)


plt.show()

