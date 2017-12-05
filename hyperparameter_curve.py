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

np.random.seed(0)
param_grid_linear = {'C': np.logspace(-4, 4, 30)}

toytest = False

if toytest:
    # Dataset
    n_samples = 100 * 1
    n_samples_low = 20 * 1
    lasso_dataset = False
    number_of_random_features = 100
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
    experiment_number = 3
    iteration = 0
    verbose = 3
    smaller_option = True
    dataset_train, dataset_test, sensible_feature_id = load_experiments(experiment_number,
                                                                        smaller_option,
                                                                        verbose)

not_fair_stats = {'error': [], 'deo': []}
fair_stats = {'error': [], 'deo': []}

# Not fair err\deo values:
for C in param_grid_linear['C']:
    estimator = svm.LinearSVC(C=C, fit_intercept=False)
    estimator.fit(dataset_train.data, dataset_train.target)
    prediction = estimator.predict(dataset_test.data)
    error = 1.0 - accuracy_score(dataset_test.target, prediction)
    subgropus_idxs = subgrups_sensible_feature_data(dataset_test.data, sensible_feature_id)
    deo = fair_tpr_from_precomputed(dataset_test.target, prediction, subgropus_idxs)
    val0 = np.min(list(deo.keys()))
    val1 = np.max(list(deo.keys()))
    not_fair_stats['error'].append(error)
    not_fair_stats['deo'].append(np.abs(deo[val0] - deo[val1]))

# Fair err\deo values:
for C in param_grid_linear['C']:
    estimator = svm.LinearSVC(C=C, fit_intercept=False)
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

plt.plot(fair_stats['error'], fair_stats['deo'], 'o', label='Fair')
plt.plot(not_fair_stats['error'], not_fair_stats['deo'], 'o', label='Not Fair')
plt.xlabel('Error')
plt.ylabel('DEO')
plt.legend()
if toytest:
    plt.title('Toy-test')
else:
    plt.title('Experiment # %d' % experiment_number)
plt.show()
