from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from sklearn import datasets, svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

from load_data import load_binary_diabetes_uci, load_heart_uci, load_breast_cancer,\
    load_adult, load_adult_race, load_adult_race_white_vs_black, laod_propublica_fairml, laod_propublica_fairml_race,\
    laod_propublica_fairml_hotencoded, load_default

from measures import fpr, fair_DEO_from_precomputed, subgrups_sensible_feature, subgrups_sensible_feature_data, tpr
from collections import namedtuple

from sklearn.model_selection import KFold


def two_step_validation_with_DEO(dataset_train, dataset_test, estimator, params, sensible_feature, scorer=accuracy_score,
                                 verbose=0, random_state=0, list_of_sensible_feature=None):

    np.random.seed(random_state)
    random_state_inner = random_state

    cv = KFold(n_splits=10, shuffle=False, random_state=random_state)
    clf = GridSearchCV(estimator=estimator, cv=cv, param_grid=params, n_jobs=1,
                       scoring=make_scorer(scorer))
    clf.fit(dataset_train.data, dataset_train.target)

    if verbose >= 2:
        print('Best score:', clf.best_score_)
        print('params:', clf.cv_results_['params'])
        print('Best C:', clf.best_estimator_.C)
        print('Best kernel:', clf.best_estimator_.kernel)
        print('Means test score:', clf.cv_results_['mean_test_score'])

    max_accuracy = clf.best_score_
    min_accepted_accuracy = max_accuracy * 0.9
    idx_accepted_accuracy = [idx for idx, val in enumerate(clf.cv_results_['mean_test_score']) if
                             val >= min_accepted_accuracy]
    params_accepted = [clf.cv_results_['params'][idx] for idx in idx_accepted_accuracy]

    if verbose >= 2:
        print('IDX of accepted accuracy:', idx_accepted_accuracy)
        print('params accepted:', params_accepted)

    inner_validation_list = []
    inner_validation_dict = {}
    for param in params_accepted:
        inner_inner_DEO = []
        cv = KFold(n_splits=10, shuffle=False, random_state=random_state_inner)
        cv_split = cv.split(dataset_train.data)
        for inner_train, inner_test in cv_split:
            if list_of_sensible_feature is None:
                dict_idxs = subgrups_sensible_feature_data(dataset_train.data[inner_test], sensible_feature)
            else:
                dict_idxs = {}
                values_of_sensible_feature = list(set(list_of_sensible_feature))
                list_of_sensible_feature = np.array(list_of_sensible_feature)
                for val in values_of_sensible_feature:
                    dict_idxs[val] = [idx for idx, s in enumerate(list_of_sensible_feature[inner_test]) if s == val]
            inner_svc = estimator
            inner_svc.set_params(**param)
            inner_svc.fit(dataset_train.data[inner_train], dataset_train.target[inner_train])
            inner_test_prediction = inner_svc.predict(dataset_train.data[inner_test])
            inner_inner_DEO.append(fair_DEO_from_precomputed(dataset_train.target[inner_test],
                                                             inner_test_prediction, dict_idxs))
        # print('Inner Inner Deo:', inner_inner_DEO)
        inner_validation_list.append(np.mean(inner_inner_DEO))
        inner_validation_dict[len(inner_validation_list) - 1] = param
        # print('Inner valid dict:', inner_validation_dict)

    if verbose >= 2:
        print('DEO validation:', inner_validation_list)
        print(inner_validation_dict)

    min_value_idx = np.argmin(inner_validation_list)  # minimum value
    min_value = np.min(inner_validation_list)
    final_best_params = inner_validation_dict[min_value_idx]
    estimator.set_params(**final_best_params)
    estimator.fit(dataset_train.data, dataset_train.target)
    # Prediction performance on test set is not as good as on train set
    pred = estimator.predict(dataset_test.data)
    score = scorer(dataset_test.target, pred)

    if verbose >= 1:
        print('Selected params:', final_best_params)
        print('with DEO:', min_value)
        print('Final model:', estimator)
        print('Test score:', score)

    return score, estimator


if __name__ == "__main__":
    print('Loading heart dataset...')
    dataset_train = load_heart_uci()
    dataset_test = load_heart_uci()
    sensible_feature = 1  # sex
    print('Different values of the sensible feature', sensible_feature, ':', set(dataset_train.data[:, sensible_feature]))
    ntrain = 8 * len(dataset_train.target) // 10
    ntest = len(dataset_train.target) - ntrain
    permutation = list(range(len(dataset_train.target)))
    np.random.shuffle(permutation)
    train_idx = permutation[:ntrain]
    test_idx = permutation[ntrain:]
    dataset_test = namedtuple('_', 'data, target')(dataset_train.data[test_idx, :], dataset_train.target[test_idx])
    dataset_train = namedtuple('_', 'data, target')(dataset_train.data[train_idx, :], dataset_train.target[train_idx])

    estimator = svm.SVC()
    Cs = np.logspace(-5, 1, 10)
    kernels = ['linear', 'rbf']

    params = [{'C': [0.1, 0.5, 1.0, 10.0, 100.0], 'kernel': ['linear']},
              {'C': [0.1, 0.5, 1.0, 10.0, 100.0], 'gamma': [0.1, 0.01], 'kernel': ['rbf']}]

    score = two_step_validation_with_DEO(dataset_train, dataset_test, estimator, params, verbose=1)
    print('SCORE:', score)
