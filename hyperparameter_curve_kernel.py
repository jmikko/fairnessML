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
from validation_method import two_step_validation_with_DEO
from collections import namedtuple
from toy_problem_lasso import toy_test_generator
from uncorrelation_nonlinear import Fair_SVM

np.random.seed(15)
param_grid = {'C': np.logspace(-6, 6, 40), 'gamma': np.logspace(-6, 6, 40)}
param_grid = {'C': np.logspace(-4, 4, 20), 'gamma': np.logspace(-4, 4, 20)}
param_grid = {'gamma': np.logspace(-4, 0, 20), 'C': np.logspace(-1, 4, 20)}

hyperlist = [(c, g) for c in param_grid['C'] for g in param_grid['gamma']]
print('Hyperlist:', hyperlist)

toytest = False
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
    # 12, 8, 2, 13, 14
    experiment_number = 13
    iteration = 0
    verbose = 3
    smaller_option = True
    dataset_train, dataset_test, sensible_feature_id = load_experiments(experiment_number,
                                                                        smaller_option,
                                                                        verbose)

not_fair_stats = {'error': [], 'deo': [], 'deo_approx': []}
fair_stats = {'error': [], 'deo': [], 'deo_approx': [], 'delta0': [], 'delta1': []}

# Not fair err\deo values:
for C, gamma in hyperlist:
    estimator = svm.SVC(C=C, kernel='rbf', gamma=gamma)
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
    print('SVM - C, gamma:', C, gamma, '- error:', error, '- EO:', deo, '- DEO:', np.abs(deo[val0] - deo[val1]), '- AppDEO:', np.abs(adeo0 - adeo1))

# Fair err\deo values:
for C, gamma in hyperlist:
    estimator = Fair_SVM(C=C, kernel='rbf', gamma=gamma, sensible_feature=sensible_feature_id)
    estimator.fit(dataset_train.data, dataset_train.target)
    prediction = estimator.predict(dataset_test.data)
    error = 1.0 - accuracy_score(dataset_test.target, prediction)
    subgropus_idxs = subgrups_sensible_feature_data(dataset_test.data, sensible_feature_id)
    deo = fair_tpr_from_precomputed(dataset_test.target, prediction, subgropus_idxs)
    val0 = np.min(list(deo.keys()))
    val1 = np.max(list(deo.keys()))
    fair_stats['error'].append(error)
    fair_stats['deo'].append(np.abs(deo[val0] - deo[val1]))

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

    adeo0lim = np.mean([np.max([-1, np.min([1, estimator.decision_function([ex])])])
                        for idx, ex in enumerate(dataset_test.data)
                        if dataset_test.target[idx] == 1 and dataset_test.data[idx][sensible_feature_id] == val0])
    adeo1lim = np.mean([np.max([-1, np.min([1, estimator.decision_function([ex])])])
                        for idx, ex in enumerate(dataset_test.data)
                        if dataset_test.target[idx] == 1 and dataset_test.data[idx][sensible_feature_id] == val1])

    delta0 = np.abs(deo[val0] - 0.5 - adeo0lim)
    delta1 = np.abs(deo[val1] - 0.5 - adeo1lim)
    fair_stats['deo_approx'].append(np.abs(adeo0 - adeo1))
    fair_stats['delta0'].append(delta0)
    fair_stats['delta1'].append(delta1)
    #  fair_stats['EO_prod'].append(deo[val0] * deo[val1])
    print('Fair-SVM - C, gamma:', C, gamma, '- error:', error, '- EO:', deo, '- DEO:', np.abs(deo[val0] - deo[val1]), '- AppDEO:', np.abs(adeo0 - adeo1),
          '\nDelta0:', delta0, 'Delta1:', delta1)


print('Not-fair STATS:', not_fair_stats)
print('Not-fair smallest error:', np.min(not_fair_stats['error']))
print('Not-fair smallest deo:', np.min(not_fair_stats['deo']))
print('Fair STATS:', fair_stats)
print('Fair smallest error:', np.min(fair_stats['error']))
print('Fair smallest deo:', np.min(fair_stats['deo']))

# besterr = np.array(fair_stats['error']).argsort()[0]
besterr = np.min(fair_stats['error'])
nearminidx = np.array([idx for idx, v in enumerate(fair_stats['error']) if v <= besterr * 1.05])
# bestallidx = nearminidx[np.argmin(fair_stats['deo'][nearminidx])]
bestallidx = nearminidx[np.array(fair_stats['deo'])[nearminidx].argsort()[:5]]
print('Best with err:', np.array(fair_stats['error'])[bestallidx])
print('Best with deo:', np.array(fair_stats['deo'])[bestallidx])
bestdelta0 = np.array(fair_stats['delta0'])[bestallidx]
bestdelta1 = np.array(fair_stats['delta1'])[bestallidx]
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
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


fig = plt.figure(1, figsize=(9, 8), dpi=80)
plt.plot(fair_stats['error'], fair_stats['deo'], 'o', markersize=15, label='Our method')
plt.plot(not_fair_stats['error'], not_fair_stats['deo'], '*', markersize=15, label='SVM')
plt.xlabel('Error')
plt.ylabel('DEO')
if toytest:
    if not lasso_dataset:
        strtitle = 'Toytest - Non-linear'
    else:
        strtitle = 'Lasso_Toytest - Non-linear'
    plt.title(strtitle)
    plt.savefig(strtitle)
else:
    strtitle = 'Experiment_%d - Non-linear' % experiment_number
    plt.title(strtitle)
    plt.savefig(strtitle)


hyperlist = np.array(hyperlist)
hypershape = (len(param_grid['C']), len(param_grid['gamma']))
for k in fair_stats:
    fair_stats[k] = np.array(fair_stats[k])
    fair_stats[k].shape = hypershape
for k in not_fair_stats:
    not_fair_stats[k] = np.array(not_fair_stats[k])
    not_fair_stats[k].shape = hypershape


cmap = 'binary'

fig = plt.figure(2, figsize=(9, 8), dpi=80)
plt.imshow(fair_stats['deo'], interpolation='bilinear', cmap=cmap, label='DEO')
plt.xlabel('log(C)')
plt.ylabel('log(Gamma)')
#plt.xticks(param_grid['C'])
#plt.yticks(param_grid['gamma'])
plt.colorbar()
#plt.legend()
plt.axes().get_xaxis().set_ticks([])
plt.axes().get_yaxis().set_ticks([])
if toytest:
    if not lasso_dataset:
        strtitle = 'Toytest - Our method - DEO'
    else:
        strtitle = 'Lasso_Toytest - Our method - DEO'
    plt.title(strtitle)
    plt.savefig(strtitle)
else:
    # strtitle = 'Experiment_%d - Our method - DEO' % experiment_number
    strtitle = 'Our method - DEO'
    plt.title(strtitle)
    plt.savefig(strtitle)


fig = plt.figure(3, figsize=(9, 8), dpi=80)
plt.imshow(fair_stats['deo_approx'], interpolation='bilinear', cmap=cmap, label='Approx DEO')
plt.xlabel('log(C)')
plt.ylabel('log(Gamma)')
plt.colorbar()
#plt.legend()
plt.axes().get_xaxis().set_ticks([])
plt.axes().get_yaxis().set_ticks([])
if toytest:
    if not lasso_dataset:
        strtitle = 'Toytest - Our method - DEO Approx'
    else:
        strtitle = 'Lasso_Toytest - Our method - DEO Approx'
    plt.title(strtitle)
    plt.savefig(strtitle)
else:
    # strtitle = 'Experiment_%d - Our method - DEO Approx' % experiment_number
    strtitle = 'Our method - DEO Approx'
    plt.title(strtitle)
    plt.savefig(strtitle)

fig = plt.figure(4, figsize=(9, 8), dpi=80)
plt.imshow(fair_stats['error'], interpolation='bilinear', cmap=cmap, label='Error')
plt.xlabel('log(C)')
plt.ylabel('log(Gamma)')
plt.colorbar()
#plt.legend()
plt.axes().get_xaxis().set_ticks([])
plt.axes().get_yaxis().set_ticks([])
if toytest:
    if not lasso_dataset:
        strtitle = 'Toytest Error - Our method'
    else:
        strtitle = 'Lasso_Toytest Error - Our method'
    plt.title(strtitle)
    plt.savefig(strtitle)
else:
    # strtitle = 'Experiment_%d Error - Our method' % experiment_number
    strtitle = 'Our method - Error'
    plt.title(strtitle)
    plt.savefig(strtitle)

fig = plt.figure(5, figsize=(9, 8), dpi=80)
plt.imshow(not_fair_stats['deo'], interpolation='bilinear', cmap=cmap, label='DEO')
plt.xlabel('log(C)')
plt.ylabel('log(Gamma)')
plt.colorbar()
#plt.legend()
plt.axes().get_xaxis().set_ticks([])
plt.axes().get_yaxis().set_ticks([])
if toytest:
    if not lasso_dataset:
        strtitle = 'Toytest - SVM - DEO'
    else:
        strtitle = 'Lasso_Toytest - SVM - DEO'
    plt.title(strtitle)
    plt.savefig(strtitle)
else:
    # strtitle = 'Experiment_%d - SVM - DEO' % experiment_number
    strtitle = 'SVM - DEO'
    plt.title(strtitle)
    plt.savefig(strtitle)

fig = plt.figure(6, figsize=(9, 8), dpi=80)
plt.imshow(not_fair_stats['deo_approx'], interpolation='bilinear', cmap=cmap, label='Approx DEO')
plt.xlabel('log(C)')
plt.ylabel('log(Gamma)')
plt.colorbar()
#plt.legend()
plt.axes().get_xaxis().set_ticks([])
plt.axes().get_yaxis().set_ticks([])
if toytest:
    if not lasso_dataset:
        strtitle = 'Toytest - SVM - DEO Approx'
    else:
        strtitle = 'Lasso_Toytest - SVM - DEO Approx'
    plt.title(strtitle)
    plt.savefig(strtitle)
else:
    # strtitle = 'Experiment_%d - SVM - DEO Approx' % experiment_number
    strtitle = 'SVM - DEO Approx'
    plt.title(strtitle)
    plt.savefig(strtitle)

fig = plt.figure(7, figsize=(9, 8), dpi=80)
plt.imshow(not_fair_stats['error'], interpolation='bilinear', cmap=cmap, label='Error')
plt.xlabel('log(C)')
plt.ylabel('log(Gamma)')
plt.colorbar()
#plt.legend()
plt.axes().get_xaxis().set_ticks([])
plt.axes().get_yaxis().set_ticks([])
if toytest:
    if not lasso_dataset:
        strtitle = 'Toytest Error - SVM'
    else:
        strtitle = 'Lasso_Toytest Error - SVM'
    plt.title(strtitle)
    plt.savefig(strtitle)
else:
    # strtitle = 'Experiment_%d Error - SVM' % experiment_number
    strtitle = 'SVM - Error'
    plt.title(strtitle)
    plt.savefig(strtitle)


plt.show()

