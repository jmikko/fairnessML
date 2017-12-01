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

# ---------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------- #


def balanced_accuracy_score(y_true, y_pred, sample_weight=None):
    """Compute the balanced accuracy
    The balanced accuracy is used in binary classification problems to deal
    with imbalanced datasets. It is defined as the arithmetic mean of
    sensitivity (true positive rate) and specificity (true negative rate),
    or the average recall obtained on either class. It is also equal to the
    ROC AUC score given binary inputs.
    The best value is 1 and the worst value is 0.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    balanced_accuracy : float.
        The average of sensitivity and specificity
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if y_type != 'binary':
        raise ValueError('Balanced accuracy is only meaningful '
                         'for binary classification problems.')
    # simply wrap the ``recall_score`` function
    return recall_score(y_true, y_pred,
                        pos_label=None,
                        average='macro',
                        sample_weight=sample_weight)

# ---------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------- #


if __name__ == '__main__':
    # Experimental settings
    experiment_number = 11
    smaller_option = False
    accuracy_balanced = False
    verbose = 3

    number_of_iterations = 30

    linear = True
    zafar = False
    not_linear = False


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

    if accuracy_balanced:
        accuracy_score = balanced_accuracy_score
    else:
        from sklearn.metrics import accuracy_score

    # ********************************************************************************************

    accuracy_train = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmK': [], 'ourK': []}
    accuracy_test = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmK': [], 'ourK': []}
    eq_opp_train = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmK': [], 'ourK': []}
    eq_opp_test = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmK': [], 'ourK': []}
    peq_opp_train = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmK': [], 'ourK': []}
    peq_opp_test = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmK': [], 'ourK': []}

    print('Experimental settings')
    print('Parameter Grid Search for Linear')
    print(param_grid_linear)
    print('Parameter Grid Search for Kernel methods')
    print(param_grid_all)
    print('Number of iterations:', number_of_iterations)

    for iteration in range(number_of_iterations):
        seed = iteration
        np.random.seed(seed)
        print('\n\n\nIteration -', iteration+1)
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
            sensible_feature = 2  # gender
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

        if verbose >= 1:
            print('Training examples:', ntrain)
            print('Test examples:', ntest)
            print('Number of features:', len(dataset_train.data[1, :]))
            values_of_sensible_feature = list(set(dataset_train.data[:, sensible_feature]))
            val0 = np.min(values_of_sensible_feature)
            val1 = np.max(values_of_sensible_feature)
            print('Examples in training in the first group:', len([el for el in dataset_train.data if el[sensible_feature] == val1]))
            print('Label True:', len([el for idx, el in enumerate(dataset_train.data) if el[sensible_feature] == val1 and dataset_train.target[idx] == 1]))
            print('Examples in training in the second group:', len([el for el in dataset_train.data if el[sensible_feature] == val0]))
            print('Label True:', len([el for idx, el in enumerate(dataset_train.data) if el[sensible_feature] == val0 and dataset_train.target[idx] == 1]))
            print('Examples in test in the first group:', len([el for el in dataset_test.data if el[sensible_feature] == val1]))
            print('Label True:', len([el for idx, el in enumerate(dataset_test.data) if el[sensible_feature] == val1 and dataset_test.target[idx] == 1]))
            print('Examples in test in the second group:', len([el for el in dataset_test.data if el[sensible_feature] == val0]))
            print('Label True:', len([el for idx, el in enumerate(dataset_test.data) if el[sensible_feature] == val0 and dataset_test.target[idx] == 1]))

        if linear:
            # Train an SVM using the training set
            print('\nGrid search for the standard Linear SVM...')
            svc = svm.SVC()
            score, best_estimator = two_step_validation_with_DEO(dataset_train, dataset_test, svc, verbose=verbose, n_jobs=n_jobs,
                                                                 sensible_feature=sensible_feature, params=param_grid_linear)

            if verbose >= 3:
                print('Y_hat:', best_estimator)
                print('Relative weight for the sensible feature:', best_estimator.coef_[0, sensible_feature])
                print('All the weights:', best_estimator.coef_[0, :])

            # Accuracy & fairness stats
            pred = best_estimator.predict(dataset_test.data)
            pred_train = best_estimator.predict(dataset_train.data)

            acctest = accuracy_score(dataset_test.target, pred)
            acctrain = accuracy_score(dataset_train.target, pred_train)
            eqopptest = equalized_odds_measure_TP(dataset_test, best_estimator, [sensible_feature], ylabel=1)
            eqopptrain = equalized_odds_measure_TP(dataset_train, best_estimator, [sensible_feature], ylabel=1)
            if verbose >= 2:
                print('Accuracy train:', acctrain)
                print('Accuracy test:', acctest)
                # Fairness measure
                print('Eq. opp. train: \n', eqopptrain)
                print('Eq. opp. test: \n', eqopptest)

            accuracy_train['svm'].append(acctrain)
            accuracy_test['svm'].append(acctest)
            eq_opp_train['svm'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] - list(eqopptrain[sensible_feature].values())[1]))
            eq_opp_test['svm'].append(np.abs(list(eqopptest[sensible_feature].values())[0] - list(eqopptest[sensible_feature].values())[1]))
            peq_opp_train['svm'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] * list(eqopptrain[sensible_feature].values())[1]))
            peq_opp_test['svm'].append(np.abs(list(eqopptest[sensible_feature].values())[0] * list(eqopptest[sensible_feature].values())[1]))

            # Hardt method
            print('\nHardt method on linear SVM...')
            algorithm = HardtMethod(dataset_train, best_estimator, sensible_feature)
            res = algorithm.fit()

            if verbose >= 2:
                if res.status == 0:
                    print('Thetas [prob. of NOT changing the prediction] y1A1, y0A1, y1A0, y0A0:', res.x[:4])
                    print('Alphas:', res.x[4:])
                else:
                    print('res.x:', res.x)
            if res.status != 0:
                print('res.status != 0:')
            else:
                theta_11, theta_01, theta_10, theta_00, alpha1, alpha2, alpha3, alpha4 = res.x
                values_of_sensible_feature = list(set(dataset_train.data[:, sensible_feature]))
                val0 = np.min(values_of_sensible_feature)
                val1 = np.max(values_of_sensible_feature)

                tmp = [1.0 if pred_train[idx] == 1 and dataset_train.data[idx, sensible_feature] == val1 else 0.0 for idx in
                       range(ntrain)]
                phi_hat_11 = np.sum(tmp) / len(tmp)
                tmp = [1.0 if pred_train[idx] == -1 and dataset_train.data[idx, sensible_feature] == val1 else 0.0 for idx in
                       range(ntrain)]
                phi_hat_01 = np.sum(tmp) / len(tmp)

                tmp = [1.0 if pred_train[idx] == 1 and dataset_train.data[idx, sensible_feature] == val0 else 0.0 for idx in
                       range(ntrain)]
                phi_hat_10 = np.sum(tmp) / len(tmp)

                tmp = [1.0 if pred_train[idx] == -1 and dataset_train.data[idx, sensible_feature] == val0 else 0.0 for idx in
                       range(ntrain)]
                phi_hat_00 = np.sum(tmp) / len(tmp)

                fair_pred_train = [float(algorithm.y_tilde(algorithm.model.predict(ex.reshape(1, -1)),
                                                           1 if ex[sensible_feature] == val1 else 0))
                                   for ex in dataset_train.data]
                fair_pred = [float(algorithm.y_tilde(algorithm.model.predict(ex.reshape(1, -1)),
                                                     1 if ex[sensible_feature] == val1 else 0))
                             for ex in dataset_test.data]
                # Accuracy
                facctrain = accuracy_score(dataset_train.target, fair_pred_train)
                facctest = accuracy_score(dataset_test.target, fair_pred)
                if verbose >= 2:
                    print('Hardt Accuracy train:', facctest)
                    print('Hardt Accuracy test:', facctrain)
                acc_Y_hat_test = accuracy_score(dataset_test.target, pred)
                acc_Y_hat_train = accuracy_score(dataset_train.target, pred_train)
                y_tilde_equals_y_hat = theta_11 * phi_hat_11 + \
                                       theta_01 * phi_hat_01 + \
                                       theta_10 * phi_hat_10 + \
                                       theta_00 * phi_hat_00

                facctestT = (2 * acc_Y_hat_test - 1) * y_tilde_equals_y_hat + 1 - acc_Y_hat_test
                facctrainT = (2 * acc_Y_hat_train - 1) * y_tilde_equals_y_hat + 1 - acc_Y_hat_train
                eqopptest = equalized_odds_measure_from_pred_TP(dataset_test, fair_pred, [sensible_feature], ylabel=1)
                eqopptrain = equalized_odds_measure_from_pred_TP(dataset_train, fair_pred_train, [sensible_feature], ylabel=1)
                if verbose >= 2:
                    print('Fair Accuracy Theoretical train:', facctrainT)
                    print('Fair Accuracy Theoretical test:', facctestT)
                    # Fairness measure
                    print('Eq. opp. train: \n', eqopptrain)  # Feature 1 is SEX
                    print('Eq. opp. test: \n', eqopptest)  # Feature 1 is SEX

            accuracy_train['hardt'].append(facctrain)
            accuracy_test['hardt'].append(facctest)
            eq_opp_train['hardt'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] - list(eqopptrain[sensible_feature].values())[1]))
            eq_opp_test['hardt'].append(np.abs(list(eqopptest[sensible_feature].values())[0] - list(eqopptest[sensible_feature].values())[1]))
            peq_opp_train['hardt'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] * list(eqopptrain[sensible_feature].values())[1]))
            peq_opp_test['hardt'].append(np.abs(list(eqopptest[sensible_feature].values())[0] * list(eqopptest[sensible_feature].values())[1]))

            # Weighted SVM
            # print('\nGrid search for the Weighted Linear SVM...')
            # svc = svm.SVC()
            # score, best_estimator = two_step_validation_with_DEO(dataset_train, dataset_test, svc,
            #                                                      sensible_feature=sensible_feature, params=param_grid_linear)
            #
            # if verbose >= 3:
            #     print('Y_hat:', best_estimator)
            #     print('Relative weight for the sensible feature:', best_estimator.coef_[0, sensible_feature])
            #     print('All the weights:', best_estimator.coef_[0, :])
            #
            # # Accuracy & fairness stats
            # pred = best_estimator.predict(dataset_test.data)
            # pred_train = best_estimator.predict(dataset_train.data)
            #
            # acctest = accuracy_score(dataset_test.target, pred)
            # acctrain = accuracy_score(dataset_train.target, pred_train)
            # eqopptest = equalized_odds_measure_TP(dataset_test, best_estimator, [sensible_feature], ylabel=1)
            # eqopptrain = equalized_odds_measure_TP(dataset_train, best_estimator, [sensible_feature], ylabel=1)
            # if verbose >= 2:
            #     print('Accuracy train:', acctrain)
            #     print('Accuracy test:', acctest)
            #     # Fairness measure
            #     print('Eq. opp. train: \n', eqopptrain)
            #     print('Eq. opp. test: \n', eqopptest)
            #
            # accuracy_train['wsvm'].append(acctrain)
            # accuracy_test['wsvm'].append(acctest)
            # eq_opp_train['wsvm'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] - list(eqopptrain[sensible_feature].values())[1]))
            # eq_opp_test['wsvm'].append(np.abs(list(eqopptest[sensible_feature].values())[0] - list(eqopptest[sensible_feature].values())[1]))
            # peq_opp_train['wsvm'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] * list(eqopptrain[sensible_feature].values())[1]))
            # peq_opp_test['wsvm'].append(np.abs(list(eqopptest[sensible_feature].values())[0] * list(eqopptest[sensible_feature].values())[1]))

            # Our uncorrelation method - Linear
            print('\nOur uncorrelation method...')
            list_of_sensible_feature_test = dataset_test.data[:, sensible_feature]
            list_of_sensible_feature_train = dataset_train.data[:, sensible_feature]
            svc = svm.SVC()
            algorithm = UncorrelationMethod(dataset_train, model=None, sensible_feature=sensible_feature)
            new_dataset_train = algorithm.new_representation(dataset_train.data)
            new_dataset_train = namedtuple('_', 'data, target')(new_dataset_train, dataset_train.target)
            new_dataset_test = algorithm.new_representation(dataset_test.data)
            new_dataset_test = namedtuple('_', 'data, target')(new_dataset_test, dataset_test.target)
            score, best_estimator = two_step_validation_with_DEO(new_dataset_train, new_dataset_test, svc, verbose=verbose, n_jobs=n_jobs,
                                                                 sensible_feature=sensible_feature, params=param_grid_linear,
                                                                 list_of_sensible_feature=[x[sensible_feature] for x in dataset_train.data])

            if verbose >= 3:
                print('Our Y:', best_estimator)

            # Accuracy
            pred = best_estimator.predict(new_dataset_test.data)
            pred_train = best_estimator.predict(new_dataset_train.data)
            facctrain = accuracy_score(new_dataset_train.target, pred_train)
            facctest = accuracy_score(new_dataset_test.target, pred)
            if verbose >= 2:
                print('Our Accuracy train:', facctest)
                print('Our Accuracy test:', facctrain)
            # Fairness measure
            eqopptrain = equalized_odds_measure_TP_from_list_of_sensfeat(new_dataset_train, best_estimator, [list_of_sensible_feature_train], ylabel=1)
            eqopptest = equalized_odds_measure_TP_from_list_of_sensfeat(new_dataset_test, best_estimator, [list_of_sensible_feature_test], ylabel=1)
            if verbose >= 2:
                print('Eq. opp. train fair: \n', eqopptrain)
                print('Eq. opp. test fair: \n', eqopptest)

            accuracy_train['our'].append(facctrain)
            accuracy_test['our'].append(facctest)
            eq_opp_train['our'].append(np.abs(list(eqopptrain[0].values())[0] - list(eqopptrain[0].values())[1]))
            eq_opp_test['our'].append(np.abs(list(eqopptest[0].values())[0] - list(eqopptest[0].values())[1]))
            peq_opp_train['our'].append(np.abs(list(eqopptrain[0].values())[0] * list(eqopptrain[0].values())[1]))
            peq_opp_test['our'].append(np.abs(list(eqopptest[0].values())[0] * list(eqopptest[0].values())[1]))

        if not_linear:
            # Train an SVM using the training set
            print('\nGrid search for the standard Kernel SVM...')
            svc = svm.SVC()
            score, best_estimator = two_step_validation_with_DEO(dataset_train, dataset_test, svc,  verbose=verbose, n_jobs=n_jobs,
                                                                 sensible_feature=sensible_feature, params=param_grid_linear)
            if verbose >= 3:
                print('Y_hat:', best_estimator)
            # Accuracy & fairness stats
            pred = best_estimator.predict(dataset_test.data)
            pred_train = best_estimator.predict(dataset_train.data)

            acctest = accuracy_score(dataset_test.target, pred)
            acctrain = accuracy_score(dataset_train.target, pred_train)
            eqopptest = equalized_odds_measure_TP(dataset_test, best_estimator, [sensible_feature], ylabel=1)
            eqopptrain = equalized_odds_measure_TP(dataset_train, best_estimator, [sensible_feature], ylabel=1)
            if verbose >= 2:
                print('Accuracy train:', acctrain)
                print('Accuracy test:', acctest)
                # Fairness measure
                print('Eq. opp. train: \n', eqopptrain)
                print('Eq. opp. test: \n', eqopptest)

            accuracy_train['svmK'].append(acctrain)
            accuracy_test['svmK'].append(acctest)
            eq_opp_train['svmK'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] - list(eqopptrain[sensible_feature].values())[1]))
            eq_opp_test['svmK'].append(np.abs(list(eqopptest[sensible_feature].values())[0] - list(eqopptest[sensible_feature].values())[1]))
            peq_opp_train['svmK'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] * list(eqopptrain[sensible_feature].values())[1]))
            peq_opp_test['svmK'].append(np.abs(list(eqopptest[sensible_feature].values())[0] * list(eqopptest[sensible_feature].values())[1]))

            # Hardt method
            print('\nHardtK method...')
            algorithm = HardtMethod(dataset_train, best_estimator, sensible_feature)
            res = algorithm.fit()

            if verbose >= 2:
                if res.status == 0:
                    print('Thetas:', res.x[:4])
                    print('Alphas:', res.x[4:])
                else:
                    print('res.x:', res.x)
            if res.status != 0:
                print('res.status != 0:')
            else:
                theta_11, theta_01, theta_10, theta_00, alpha1, alpha2, alpha3, alpha4 = res.x
                values_of_sensible_feature = list(set(dataset_train.data[:, sensible_feature]))
                val0 = np.min(values_of_sensible_feature)
                val1 = np.max(values_of_sensible_feature)

                tmp = [1.0 if pred_train[idx] == 1 and dataset_train.data[idx, sensible_feature] == val1 else 0.0 for idx in
                       range(ntrain)]
                phi_hat_11 = np.sum(tmp) / len(tmp)
                tmp = [1.0 if pred_train[idx] == -1 and dataset_train.data[idx, sensible_feature] == val1 else 0.0 for idx in
                       range(ntrain)]
                phi_hat_01 = np.sum(tmp) / len(tmp)

                tmp = [1.0 if pred_train[idx] == 1 and dataset_train.data[idx, sensible_feature] == val0 else 0.0 for idx in
                       range(ntrain)]
                phi_hat_10 = np.sum(tmp) / len(tmp)

                tmp = [1.0 if pred_train[idx] == -1 and dataset_train.data[idx, sensible_feature] == val0 else 0.0 for idx in
                       range(ntrain)]
                phi_hat_00 = np.sum(tmp) / len(tmp)

                fair_pred_train = [float(algorithm.y_tilde(algorithm.model.predict(ex.reshape(1, -1)),
                                                           1 if ex[sensible_feature] == val1 else 0))
                                   for ex in dataset_train.data]
                fair_pred = [float(algorithm.y_tilde(algorithm.model.predict(ex.reshape(1, -1)),
                                                     1 if ex[sensible_feature] == val1 else 0))
                             for ex in dataset_test.data]
                # Accuracy
                facctrain = accuracy_score(dataset_train.target, fair_pred_train)
                facctest = accuracy_score(dataset_test.target, fair_pred)
                if verbose >= 2:
                    print('HardtK Accuracy train:', facctest)
                    print('HardtK Accuracy test:', facctrain)
                acc_Y_hat_test = accuracy_score(dataset_test.target, pred)
                acc_Y_hat_train = accuracy_score(dataset_train.target, pred_train)
                y_tilde_equals_y_hat = theta_11 * phi_hat_11 + \
                                       theta_01 * phi_hat_01 + \
                                       theta_10 * phi_hat_10 + \
                                       theta_00 * phi_hat_00

                facctestT = (2 * acc_Y_hat_test - 1) * y_tilde_equals_y_hat + 1 - acc_Y_hat_test
                facctrainT = (2 * acc_Y_hat_train - 1) * y_tilde_equals_y_hat + 1 - acc_Y_hat_train
                eqopptest = equalized_odds_measure_from_pred_TP(dataset_test, fair_pred, [sensible_feature], ylabel=1)
                eqopptrain = equalized_odds_measure_from_pred_TP(dataset_train, fair_pred_train, [sensible_feature], ylabel=1)
                if verbose >= 2:
                    print('Fair Accuracy Theoretical train:', facctrainT)
                    print('Fair Accuracy Theoretical test:', facctestT)
                    # Fairness measure
                    print('Eq. opp. train: \n', eqopptrain)  # Feature 1 is SEX
                    print('Eq. opp. test: \n', eqopptest)  # Feature 1 is SEX

            accuracy_train['hardtK'].append(facctrain)
            accuracy_test['hardtK'].append(facctest)
            eq_opp_train['hardtK'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] - list(eqopptrain[sensible_feature].values())[1]))
            eq_opp_test['hardtK'].append(np.abs(list(eqopptest[sensible_feature].values())[0] - list(eqopptest[sensible_feature].values())[1]))
            peq_opp_train['hardtK'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] * list(eqopptrain[sensible_feature].values())[1]))
            peq_opp_test['hardtK'].append(np.abs(list(eqopptest[sensible_feature].values())[0] * list(eqopptest[sensible_feature].values())[1]))


            # Our uncorrelation method - Kernel
            print('\nOur uncorrelation method with kernels...')
            list_of_sensible_feature_test = dataset_test.data[:, sensible_feature]
            list_of_sensible_feature_train = dataset_train.data[:, sensible_feature]

            svc = Fair_SVM(sensible_feature=sensible_feature)
            score, best_estimator = two_step_validation_with_DEO(dataset_train, dataset_test, svc,  verbose=verbose, n_jobs=n_jobs,
                                                                 sensible_feature=sensible_feature, params=param_grid_linear)


            if verbose >= 3:
                print('Our Y:', best_estimator)

            # Accuracy
            facctest = best_estimator.score(dataset_test.data, dataset_test.target)
            facctrain = best_estimator.score(dataset_train.data, dataset_train.target)
            if verbose >= 2:
                print('Our Accuracy train:', facctest)
                print('Our Accuracy test:', facctrain)
            # Fairness measure
            eqopptrain = equalized_odds_measure_TP_from_list_of_sensfeat(dataset_train, best_estimator, [list_of_sensible_feature_train], ylabel=1)
            eqopptest = equalized_odds_measure_TP_from_list_of_sensfeat(dataset_test, best_estimator, [list_of_sensible_feature_test], ylabel=1)
            if verbose >= 2:
                print('Eq. opp. train fair: \n', eqopptrain)
                print('Eq. opp. test fair: \n', eqopptest)

            accuracy_train['ourK'].append(facctrain)
            accuracy_test['ourK'].append(facctest)
            eq_opp_train['ourK'].append(np.abs(list(eqopptrain[0].values())[0] - list(eqopptrain[0].values())[1]))
            eq_opp_test['ourK'].append(np.abs(list(eqopptest[0].values())[0] - list(eqopptest[0].values())[1]))
            peq_opp_train['ourK'].append(np.abs(list(eqopptrain[0].values())[0] * list(eqopptrain[0].values())[1]))
            peq_opp_test['ourK'].append(np.abs(list(eqopptest[0].values())[0] * list(eqopptest[0].values())[1]))

        if zafar:
            # Zafar
            sys.path.insert(0, './zafar_methods/fair_classification/')
            # from generate_synthetic_data import *
            import utils as ut
            import funcs_disp_mist as fdm
            import loss_funcs as lf  # loss funcs that can be optimized subject to various constraints

            print('\nZafar method...')
            X, y, x_control = dataset_train.data, dataset_train.target, {"s1": dataset_train.data[:, sensible_feature]}
            sensitive_attrs = x_control.keys()
            x_train, y_train, x_control_train, x_test, y_test, x_control_test = dataset_train.data[:, :sensible_feature] + dataset_train.data[:, sensible_feature+1:], dataset_train.target, {"s1": dataset_train.data[:, sensible_feature]}, \
                                                                                dataset_test.data[:, :sensible_feature] + dataset_test.data[:, sensible_feature + 1:], dataset_test.target, {"s1": dataset_test.data[:, sensible_feature]}
            cons_params = None  # constraint parameters, will use them later
            loss_function = "logreg"  # perform the experiments with logistic regression
            EPS = 1e-4
            def train_test_classifier():
                w = fdm.train_model_disp_mist(x_train, y_train, x_control_train, loss_function, EPS, cons_params)

                train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(
                    w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs)

                # accuracy and FPR are for the test because we need of for plotting
                # the covariance is for train, because we need it for setting the thresholds
                return w, train_score, test_score, s_attr_to_fp_fn_test, s_attr_to_fp_fn_train, cov_all_train

            w_uncons, acc_untrain, acc_uncons, s_attr_to_fp_fn_test_uncons, s_attr_to_fp_fn_test_uncons, cov_all_train_uncons = train_test_classifier()
            it = 0.05
            mult_range = np.arange(1.0, 0.0 - it, -it).tolist()
            acc_arr = []
            fpr_per_group = {0: [], 1: []}
            fnr_per_group = {0: [], 1: []}
            cons_type = 1  # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
            tau = 5.0
            mu = 1.2

            for m in mult_range:
                sensitive_attrs_to_cov_thresh = deepcopy(cov_all_train_uncons)
                for s_attr in sensitive_attrs_to_cov_thresh.keys():
                    for cov_type in sensitive_attrs_to_cov_thresh[s_attr].keys():
                        for s_val in sensitive_attrs_to_cov_thresh[s_attr][cov_type]:
                            sensitive_attrs_to_cov_thresh[s_attr][cov_type][s_val] *= m

                cons_params = {"cons_type": cons_type,
                               "tau": tau,
                               "mu": mu,
                               "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

                w_cons, acc_train, acc_cons, s_attr_to_fp_fn_test_cons, s_attr_to_fp_fn_train_cons, cov_all_train_cons = train_test_classifier()
                fpr_per_group[0].append(s_attr_to_fp_fn_test_cons["s1"][0.0]["fpr"])
                fpr_per_group[1].append(s_attr_to_fp_fn_test_cons["s1"][1.0]["fpr"])
                fnr_per_group[0].append(s_attr_to_fp_fn_test_cons["s1"][0.0]["fnr"])
                fnr_per_group[1].append(s_attr_to_fp_fn_test_cons["s1"][1.0]["fnr"])
                acc_arr.append(acc_cons)

            accuracy_train['zafar'].append(acc_train)
            accuracy_test['zafar'].append(acc_cons)
            eq_opp_train['zafar'].append(np.abs(s_attr_to_fp_fn_train_cons["s1"][0.0]["fpr"] - s_attr_to_fp_fn_train_cons["s1"][1.0]["fpr"]))
            eq_opp_test['zafar'].append(np.abs(s_attr_to_fp_fn_test_cons["s1"][0.0]["fpr"] - s_attr_to_fp_fn_test_cons["s1"][1.0]["fpr"]))
            peq_opp_train['zafar'].append(np.abs(s_attr_to_fp_fn_train_cons["s1"][0.0]["fpr"] * s_attr_to_fp_fn_train_cons["s1"][1.0]["fpr"]))
            peq_opp_test['zafar'].append(np.abs(s_attr_to_fp_fn_test_cons["s1"][0.0]["fpr"] * s_attr_to_fp_fn_test_cons["s1"][1.0]["fpr"]))

        if verbose >= 1 and iteration != number_of_iterations - 1:
            print('\n\nStats at iteration', iteration + 1)
            print('Method \t Accuracy on train \t Accuracy on test \t Diff.Eq.Opp.train \t Diff.Eq.Opp.test \t Prod.Eq.Opp.train \t Prod.Eq.Opp.test')
            if linear:
                print('SVM \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                      % (np.mean(accuracy_train['svm']), np.std(accuracy_train['svm']),
                         np.mean(accuracy_test['svm']), np.std(accuracy_test['svm']),
                         np.mean(eq_opp_train['svm']), np.std(eq_opp_train['svm']),
                         np.mean(eq_opp_test['svm']), np.std(eq_opp_test['svm']),
                         np.mean(peq_opp_train['svm']), np.std(peq_opp_train['svm']),
                         np.mean(peq_opp_test['svm']), np.std(peq_opp_test['svm'])))
                print('Hardt \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                      % (np.mean(accuracy_train['hardt']), np.std(accuracy_train['hardt']),
                         np.mean(accuracy_test['hardt']), np.std(accuracy_test['hardt']),
                         np.mean(eq_opp_train['hardt']), np.std(eq_opp_train['hardt']),
                         np.mean(eq_opp_test['hardt']), np.std(eq_opp_test['hardt']),
                         np.mean(peq_opp_train['hardt']), np.std(peq_opp_train['hardt']),
                         np.mean(peq_opp_test['hardt']), np.std(peq_opp_test['hardt'])))
                print('Our \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                      % (np.mean(accuracy_train['our']), np.std(accuracy_train['our']),
                         np.mean(accuracy_test['our']), np.std(accuracy_test['our']),
                         np.mean(eq_opp_train['our']), np.std(eq_opp_train['our']),
                         np.mean(eq_opp_test['our']), np.std(eq_opp_test['our']),
                         np.mean(peq_opp_train['our']), np.std(peq_opp_train['our']),
                         np.mean(peq_opp_test['our']), np.std(peq_opp_test['our'])))
            if zafar:
                print('Zafar \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                      % (np.mean(accuracy_train['zafar']), np.std(accuracy_train['zafar']),
                         np.mean(accuracy_test['zafar']), np.std(accuracy_test['zafar']),
                         np.mean(eq_opp_train['zafar']), np.std(eq_opp_train['zafar']),
                         np.mean(eq_opp_test['zafar']), np.std(eq_opp_test['zafar']),
                         np.mean(peq_opp_train['zafar']), np.std(peq_opp_train['zafar']),
                         np.mean(peq_opp_test['zafar']), np.std(peq_opp_test['zafar'])))
            if not_linear:
                print('SVMK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                      % (np.mean(accuracy_train['svmK']), np.std(accuracy_train['svmK']),
                         np.mean(accuracy_test['svmK']), np.std(accuracy_test['svmK']),
                         np.mean(eq_opp_train['svmK']), np.std(eq_opp_train['svmK']),
                         np.mean(eq_opp_test['svmK']), np.std(eq_opp_test['svmK']),
                         np.mean(peq_opp_train['svmK']), np.std(peq_opp_train['svmK']),
                         np.mean(peq_opp_test['svmK']), np.std(peq_opp_test['svmK'])))
                print('HardtK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                      % (np.mean(accuracy_train['hardtK']), np.std(accuracy_train['hardtK']),
                         np.mean(accuracy_test['hardtK']), np.std(accuracy_test['hardtK']),
                         np.mean(eq_opp_train['hardtK']), np.std(eq_opp_train['hardtK']),
                         np.mean(eq_opp_test['hardtK']), np.std(eq_opp_test['hardtK']),
                         np.mean(peq_opp_train['hardtK']), np.std(peq_opp_train['hardtK']),
                         np.mean(peq_opp_test['hardtK']), np.std(peq_opp_test['hardtK'])))
                print('OurK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                      % (np.mean(accuracy_train['ourK']), np.std(accuracy_train['ourK']),
                         np.mean(accuracy_test['ourK']), np.std(accuracy_test['ourK']),
                         np.mean(eq_opp_train['ourK']), np.std(eq_opp_train['ourK']),
                         np.mean(eq_opp_test['ourK']), np.std(eq_opp_test['ourK']),
                         np.mean(peq_opp_train['ourK']), np.std(peq_opp_train['ourK']),
                         np.mean(peq_opp_test['ourK']), np.std(peq_opp_test['ourK'])))

    print('\n\n\n\nFinal stats (after', iteration+1, 'iterations)')
    print('Method \t Accuracy on train \t Accuracy on test \t Diff.Eq.Opp.train \t Diff.Eq.Opp.test \t Prod.Eq.Opp.train \t Prod.Eq.Opp.test')
    if linear:
        print('SVM \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['svm']), np.std(accuracy_train['svm']),
                 np.mean(accuracy_test['svm']), np.std(accuracy_test['svm']),
                 np.mean(eq_opp_train['svm']), np.std(eq_opp_train['svm']),
                 np.mean(eq_opp_test['svm']), np.std(eq_opp_test['svm']),
                 np.mean(peq_opp_train['svm']), np.std(peq_opp_train['svm']),
                 np.mean(peq_opp_test['svm']), np.std(peq_opp_test['svm'])))
        print('Hardt \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['hardt']), np.std(accuracy_train['hardt']),
                 np.mean(accuracy_test['hardt']), np.std(accuracy_test['hardt']),
                 np.mean(eq_opp_train['hardt']), np.std(eq_opp_train['hardt']),
                 np.mean(eq_opp_test['hardt']), np.std(eq_opp_test['hardt']),
                 np.mean(peq_opp_train['hardt']), np.std(peq_opp_train['hardt']),
                 np.mean(peq_opp_test['hardt']), np.std(peq_opp_test['hardt'])))
        print('Our \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['our']), np.std(accuracy_train['our']),
                 np.mean(accuracy_test['our']), np.std(accuracy_test['our']),
                 np.mean(eq_opp_train['our']), np.std(eq_opp_train['our']),
                 np.mean(eq_opp_test['our']), np.std(eq_opp_test['our']),
                 np.mean(peq_opp_train['our']), np.std(peq_opp_train['our']),
                 np.mean(peq_opp_test['our']), np.std(peq_opp_test['our'])))
    if zafar:
        print('Zafar \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['zafar']), np.std(accuracy_train['zafar']),
                 np.mean(accuracy_test['zafar']), np.std(accuracy_test['zafar']),
                 np.mean(eq_opp_train['zafar']), np.std(eq_opp_train['zafar']),
                 np.mean(eq_opp_test['zafar']), np.std(eq_opp_test['zafar']),
                 np.mean(peq_opp_train['zafar']), np.std(peq_opp_train['zafar']),
                 np.mean(peq_opp_test['zafar']), np.std(peq_opp_test['zafar'])))
    if not_linear:
        print('SVMK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['svmK']), np.std(accuracy_train['svmK']),
                 np.mean(accuracy_test['svmK']), np.std(accuracy_test['svmK']),
                 np.mean(eq_opp_train['svmK']), np.std(eq_opp_train['svmK']),
                 np.mean(eq_opp_test['svmK']), np.std(eq_opp_test['svmK']),
                 np.mean(peq_opp_train['svmK']), np.std(peq_opp_train['svmK']),
                 np.mean(peq_opp_test['svmK']), np.std(peq_opp_test['svmK'])))
        print('HardtK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['hardtK']), np.std(accuracy_train['hardtK']),
                 np.mean(accuracy_test['hardtK']), np.std(accuracy_test['hardtK']),
                 np.mean(eq_opp_train['hardtK']), np.std(eq_opp_train['hardtK']),
                 np.mean(eq_opp_test['hardtK']), np.std(eq_opp_test['hardtK']),
                 np.mean(peq_opp_train['hardtK']), np.std(peq_opp_train['hardtK']),
                 np.mean(peq_opp_test['hardtK']), np.std(peq_opp_test['hardtK'])))
        print('OurK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['ourK']), np.std(accuracy_train['ourK']),
                 np.mean(accuracy_test['ourK']), np.std(accuracy_test['ourK']),
                 np.mean(eq_opp_train['ourK']), np.std(eq_opp_train['ourK']),
                 np.mean(eq_opp_test['ourK']), np.std(eq_opp_test['ourK']),
                 np.mean(peq_opp_train['ourK']), np.std(peq_opp_train['ourK']),
                 np.mean(peq_opp_test['ourK']), np.std(peq_opp_test['ourK'])))

