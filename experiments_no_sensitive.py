from load_data import load_experiments
from sklearn import svm
import numpy as np
from measures import equalized_odds_measure_FP, equalized_odds_measure_from_pred_TP, equalized_odds_measure_TP_from_list_of_sensfeat
import matplotlib.pyplot as plt
from measures import equalized_odds_measure_TP_no_sensitive as equalized_odds_measure_TP
from sklearn.model_selection import GridSearchCV
from scipy.optimize import linprog
from hardt import gamma_y_hat, HardtMethod
from scipy.spatial import ConvexHull
from uncorrelation_no_sensitive import UncorrelationMethod_no_sensitive as UncorrelationMethod
from uncorrelation_nonlinear_no_sensitive import Fair_SVM_no_sensitive as Fair_SVM
from uncorrelation_nonlinear_epsilon import Fair_SVM_eps
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
from sklearn.model_selection import KFold

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
    # 12, 8, 2, 13, 14
    experiment_number = 2
    smaller_option = False
    accuracy_balanced = False
    verbose = 3

    number_of_iterations = 3

    linear = True
    zafar = True
    not_linear = False
    our_epsilon = False

    if our_epsilon:
        epsilon = 0.5
        print('Epsilon for our method:', epsilon)


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

    accuracy_train = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmV': [], 'svmK': [], 'svmVK': [], 'ourK': [], 'ourKeps': []}
    accuracy_test = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmV': [], 'svmK': [], 'svmVK': [], 'ourK': [], 'ourKeps': []}
    eq_opp_train = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmV': [], 'svmK': [], 'svmVK': [], 'ourK': [], 'ourKeps': []}
    eq_opp_test = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmV': [], 'svmK': [], 'svmVK': [], 'ourK': [], 'ourKeps': []}
    peq_opp_train = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmV': [], 'svmK': [], 'svmVK': [], 'ourK': [], 'ourKeps': []}
    peq_opp_test = {'hardt': [], 'hardtK': [], 'our': [], 'zafar': [], 'svm': [], 'svmV': [], 'svmK': [], 'svmVK': [], 'ourK': [], 'ourKeps': []}

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
        dataset_train, dataset_test, sensible_feature = load_experiments(experiment_number, smaller_option, verbose)
        ntrain = len(dataset_train.target)

        dataset_test_no_sensitive = np.delete(dataset_test.data, sensible_feature, 1)
        dataset_train_no_sensitive = np.delete(dataset_train.data, sensible_feature, 1)

        if linear:
            # Train an SVM using the training set
            print('\nGrid search for the standard Linear SVM with standard validation...')
            svc = svm.SVC()
            cv = KFold(n_splits=5, shuffle=False, random_state=seed)
            clf = GridSearchCV(estimator=svc, cv=cv, param_grid=param_grid_linear, n_jobs=n_jobs,
                               scoring=make_scorer(accuracy_score))
            clf.fit(dataset_train_no_sensitive, dataset_train.target)
            best_estimator = clf.best_estimator_

            if verbose >= 3:
                print('Y_hat:', best_estimator)
                #print('Relative weight for the sensible feature:', best_estimator.coef_[0, sensible_feature])
                print('All the weights:', best_estimator.coef_[0, :])

            # Accuracy & fairness stats
            pred = best_estimator.predict(dataset_test_no_sensitive)
            pred_train = best_estimator.predict(dataset_train_no_sensitive)

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

            accuracy_train['svmV'].append(acctrain)
            accuracy_test['svmV'].append(acctest)
            eq_opp_train['svmV'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] - list(eqopptrain[sensible_feature].values())[1]))
            eq_opp_test['svmV'].append(np.abs(list(eqopptest[sensible_feature].values())[0] - list(eqopptest[sensible_feature].values())[1]))
            peq_opp_train['svmV'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] * list(eqopptrain[sensible_feature].values())[1]))
            peq_opp_test['svmV'].append(np.abs(list(eqopptest[sensible_feature].values())[0] * list(eqopptest[sensible_feature].values())[1]))

            # Train an SVM using the training set
            print('\nGrid search for the standard Linear SVM...')
            svc = svm.SVC()
            score, best_estimator = two_step_validation_with_DEO(dataset_train, dataset_test, svc, verbose=verbose, n_jobs=n_jobs,
                                                                 sensible_feature=sensible_feature, params=param_grid_linear,
                                                                 no_sensitive=True)

            if verbose >= 3:
                print('Y_hat:', best_estimator)
                #print('Relative weight for the sensible feature:', best_estimator.coef_[0, sensible_feature])
                print('All the weights:', best_estimator.coef_[0, :])

            # Accuracy & fairness stats
            pred = best_estimator.predict(dataset_test_no_sensitive)
            pred_train = best_estimator.predict(dataset_train_no_sensitive)

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
            print('\nHardt method impossible...')
            accuracy_train['hardt'].append(-1)
            accuracy_test['hardt'].append(-1)
            eq_opp_train['hardt'].append(-1)
            eq_opp_test['hardt'].append(-1)
            peq_opp_train['hardt'].append(-1)
            peq_opp_test['hardt'].append(-1)


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
            print('\nGrid search for the standard Kernel SVM with standard validation...')
            svc = svm.SVC()
            cv = KFold(n_splits=5, shuffle=False, random_state=seed)
            clf = GridSearchCV(estimator=svc, cv=cv, param_grid=param_grid_all, n_jobs=n_jobs,
                               scoring=make_scorer(accuracy_score))
            clf.fit(dataset_train_no_sensitive, dataset_train.target)
            best_estimator = clf.best_estimator_

            if verbose >= 3:
                print('Y_hat:', best_estimator)

            # Accuracy & fairness stats
            pred = best_estimator.predict(dataset_test_no_sensitive)
            pred_train = best_estimator.predict(dataset_train_no_sensitive)

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

            accuracy_train['svmVK'].append(acctrain)
            accuracy_test['svmVK'].append(acctest)
            eq_opp_train['svmVK'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] - list(eqopptrain[sensible_feature].values())[1]))
            eq_opp_test['svmVK'].append(np.abs(list(eqopptest[sensible_feature].values())[0] - list(eqopptest[sensible_feature].values())[1]))
            peq_opp_train['svmVK'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] * list(eqopptrain[sensible_feature].values())[1]))
            peq_opp_test['svmVK'].append(np.abs(list(eqopptest[sensible_feature].values())[0] * list(eqopptest[sensible_feature].values())[1]))

            # Train an SVM using the training set
            print('\nGrid search for the standard Kernel SVM...')
            svc = svm.SVC()
            score, best_estimator = two_step_validation_with_DEO(dataset_train, dataset_test, svc,  verbose=verbose, n_jobs=n_jobs,
                                                                 sensible_feature=sensible_feature, params=param_grid_linear, no_sensitive=True)
            if verbose >= 3:
                print('Y_hat:', best_estimator)
            # Accuracy & fairness stats
            pred = best_estimator.predict(dataset_test_no_sensitive)
            pred_train = best_estimator.predict(dataset_train_no_sensitive)

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
            print('\nHardtK method impossible...')

            accuracy_train['hardtK'].append(-1)
            accuracy_test['hardtK'].append(-1)
            eq_opp_train['hardtK'].append(-1)
            eq_opp_test['hardtK'].append(-1)
            peq_opp_train['hardtK'].append(-1)
            peq_opp_test['hardtK'].append(-1)


            # Our uncorrelation method - Kernel
            print('\nOur uncorrelation method with kernels...')
            list_of_sensible_feature_test = dataset_test.data[:, sensible_feature]
            list_of_sensible_feature_train = dataset_train.data[:, sensible_feature]

            svc = Fair_SVM(sensible_feature=sensible_feature)
            score, best_estimator = two_step_validation_with_DEO(dataset_train, dataset_test, svc,  verbose=verbose, n_jobs=n_jobs,
                                                                 sensible_feature=sensible_feature, params=param_grid_all)


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

        if our_epsilon:
            # Our uncorrelation method - Kernel with epsilon
            print('\nOur uncorrelation method with epsilon with kernels...')
            list_of_sensible_feature_test = dataset_test.data[:, sensible_feature]
            list_of_sensible_feature_train = dataset_train.data[:, sensible_feature]

            svc = Fair_SVM_eps(sensible_feature=sensible_feature, epsilon=epsilon)
            score, best_estimator = two_step_validation_with_DEO(dataset_train, dataset_test, svc, verbose=verbose,
                                                                 n_jobs=n_jobs,
                                                                 sensible_feature=sensible_feature,
                                                                 params=param_grid_all)

            if verbose >= 3:
                print('Our Y:', best_estimator)

            # Accuracy
            facctest = best_estimator.score(dataset_test.data, dataset_test.target)
            facctrain = best_estimator.score(dataset_train.data, dataset_train.target)
            if verbose >= 2:
                print('Our Accuracy train:', facctest)
                print('Our Accuracy test:', facctrain)
            # Fairness measure
            eqopptrain = equalized_odds_measure_TP_from_list_of_sensfeat(dataset_train, best_estimator,
                                                                         [list_of_sensible_feature_train], ylabel=1)
            eqopptest = equalized_odds_measure_TP_from_list_of_sensfeat(dataset_test, best_estimator,
                                                                        [list_of_sensible_feature_test], ylabel=1)
            if verbose >= 2:
                print('Eq. opp. train fair: \n', eqopptrain)
                print('Eq. opp. test fair: \n', eqopptest)

            accuracy_train['ourKeps'].append(facctrain)
            accuracy_test['ourKeps'].append(facctest)
            eq_opp_train['ourKeps'].append(np.abs(list(eqopptrain[0].values())[0] - list(eqopptrain[0].values())[1]))
            eq_opp_test['ourKeps'].append(np.abs(list(eqopptest[0].values())[0] - list(eqopptest[0].values())[1]))
            peq_opp_train['ourKeps'].append(np.abs(list(eqopptrain[0].values())[0] * list(eqopptrain[0].values())[1]))
            peq_opp_test['ourKeps'].append(np.abs(list(eqopptest[0].values())[0] * list(eqopptest[0].values())[1]))


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
            val0 = np.min(dataset_train.data[:, sensible_feature])
            val1 = np.max(dataset_train.data[:, sensible_feature])
            new_train_sensitive = np.array([0 if valx == val0 else 1 for valx in dataset_train.data[:, sensible_feature]])
            new_test_sensitive = np.array([0 if valx == val0 else 1 for valx in dataset_test.data[:, sensible_feature]])
            x_train, y_train, x_control_train, x_test, y_test, x_control_test = np.hstack((dataset_train.data[:, :sensible_feature], dataset_train.data[:, sensible_feature+1:])), dataset_train.target, {"s1": new_train_sensitive},\
                                                                                np.hstack((dataset_test.data[:, :sensible_feature], dataset_test.data[:, sensible_feature + 1:])), dataset_test.target, {"s1": new_test_sensitive}
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

            if verbose >= 2:
                print('Zafar list of values:')
                print('For Test:', s_attr_to_fp_fn_test_cons)
                print('For Train:', s_attr_to_fp_fn_train_cons)

        if verbose >= 1 and iteration != number_of_iterations - 1:
            print('\n\nStats at iteration', iteration + 1)
            print('Method \t Accuracy on train \t Accuracy on test \t Diff.Eq.Opp.train \t Diff.Eq.Opp.test \t Prod.Eq.Opp.train \t Prod.Eq.Opp.test')
            if linear:
                print('SVMV \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                      % (np.mean(accuracy_train['svmV']), np.std(accuracy_train['svmV']),
                         np.mean(accuracy_test['svmV']), np.std(accuracy_test['svmV']),
                         np.mean(eq_opp_train['svmV']), np.std(eq_opp_train['svmV']),
                         np.mean(eq_opp_test['svmV']), np.std(eq_opp_test['svmV']),
                         np.mean(peq_opp_train['svmV']), np.std(peq_opp_train['svmV']),
                         np.mean(peq_opp_test['svmV']), np.std(peq_opp_test['svmV'])))
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
                print('SVMVK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                      % (np.mean(accuracy_train['svmVK']), np.std(accuracy_train['svmVK']),
                         np.mean(accuracy_test['svmVK']), np.std(accuracy_test['svmVK']),
                         np.mean(eq_opp_train['svmVK']), np.std(eq_opp_train['svmVK']),
                         np.mean(eq_opp_test['svmVK']), np.std(eq_opp_test['svmVK']),
                         np.mean(peq_opp_train['svmVK']), np.std(peq_opp_train['svmVK']),
                         np.mean(peq_opp_test['svmVK']), np.std(peq_opp_test['svmVK'])))
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
            if our_epsilon:
                print('OurKeps \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                      % (np.mean(accuracy_train['ourKeps']), np.std(accuracy_train['ourKeps']),
                         np.mean(accuracy_test['ourKeps']), np.std(accuracy_test['ourKeps']),
                         np.mean(eq_opp_train['ourKeps']), np.std(eq_opp_train['ourKeps']),
                         np.mean(eq_opp_test['ourKeps']), np.std(eq_opp_test['ourKeps']),
                         np.mean(peq_opp_train['ourKeps']), np.std(peq_opp_train['ourKeps']),
                         np.mean(peq_opp_test['ourKeps']), np.std(peq_opp_test['ourKeps'])))

    print('\n\n\n\nFinal stats (after', iteration+1, 'iterations)')
    print('Method \t Accuracy on train \t Accuracy on test \t Diff.Eq.Opp.train \t Diff.Eq.Opp.test \t Prod.Eq.Opp.train \t Prod.Eq.Opp.test')
    if linear:
        print('SVMV \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['svmV']), np.std(accuracy_train['svmV']),
                 np.mean(accuracy_test['svmV']), np.std(accuracy_test['svmV']),
                 np.mean(eq_opp_train['svmV']), np.std(eq_opp_train['svmV']),
                 np.mean(eq_opp_test['svmV']), np.std(eq_opp_test['svmV']),
                 np.mean(peq_opp_train['svmV']), np.std(peq_opp_train['svmV']),
                 np.mean(peq_opp_test['svmV']), np.std(peq_opp_test['svmV'])))
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
        print('SVMVK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['svmVK']), np.std(accuracy_train['svmVK']),
                 np.mean(accuracy_test['svmVK']), np.std(accuracy_test['svmVK']),
                 np.mean(eq_opp_train['svmVK']), np.std(eq_opp_train['svmVK']),
                 np.mean(eq_opp_test['svmVK']), np.std(eq_opp_test['svmVK']),
                 np.mean(peq_opp_train['svmVK']), np.std(peq_opp_train['svmVK']),
                 np.mean(peq_opp_test['svmVK']), np.std(peq_opp_test['svmVK'])))
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
    if our_epsilon:
        print('OurKeps \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['ourKeps']), np.std(accuracy_train['ourKeps']),
                 np.mean(accuracy_test['ourKeps']), np.std(accuracy_test['ourKeps']),
                 np.mean(eq_opp_train['ourKeps']), np.std(eq_opp_train['ourKeps']),
                 np.mean(eq_opp_test['ourKeps']), np.std(eq_opp_test['ourKeps']),
                 np.mean(peq_opp_train['ourKeps']), np.std(peq_opp_train['ourKeps']),
                 np.mean(peq_opp_test['ourKeps']), np.std(peq_opp_test['ourKeps'])))

