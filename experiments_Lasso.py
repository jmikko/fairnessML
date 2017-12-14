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
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso


class LassoC(Lasso):
    def predict(self, X):
        return np.sign(np.sign(super().predict(X)) + 0.1)

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
    experiment_number = 12
    smaller_option = False
    accuracy_balanced = False
    verbose = 3

    near_zero = 1e-8
    max_iterations = 10000
    number_of_iterations = 10

    grid_search_complete = True
    n_jobs = 2
    if grid_search_complete:
        param_grid_linear = [
            {'alpha': np.logspace(-2, 4, 15)}
        ]
    else:
        print('---> No grid search performed! <---')
        param_grid_linear = [{'alpha': [1.0]}]

    if smaller_option:
        print('---> A smaller dataset could be loaded <---')

    if accuracy_balanced:
        accuracy_score = balanced_accuracy_score
    else:
        from sklearn.metrics import accuracy_score

    # ********************************************************************************************

    accuracy_train = {'hardt': [], 'our': [], 'svm': [], 'svmV': []}
    accuracy_test = {'hardt': [], 'our': [], 'svm': [], 'svmV': []}
    eq_opp_train = {'hardt': [], 'our': [], 'svm': [], 'svmV': []}
    eq_opp_test = {'hardt': [], 'our': [], 'svm': [], 'svmV': []}
    peq_opp_train = {'hardt': [], 'our': [], 'svm': [], 'svmV': []}
    peq_opp_test = {'hardt': [], 'our': [], 'svm': [], 'svmV': []}

    number_selected_features = {'hardt': [], 'our': [], 'svm': [], 'svmV': []}

    print('Experimental settings')
    print('Parameter Grid Search for Linear')
    print(param_grid_linear)
    print('Number of iterations:', number_of_iterations)

    for iteration in range(number_of_iterations):
        seed = iteration
        np.random.seed(seed)
        print('\n\n\nIteration -', iteration+1)
        dataset_train, dataset_test, sensible_feature = load_experiments(experiment_number, smaller_option, verbose)
        ntrain = len(dataset_train.target)

        if True:
            # Train an SVM using the training set
            print('\nGrid search for the standard Linear SVM with standard validation...')
            lasso = LassoC(max_iter=max_iterations)
            cv = KFold(n_splits=5, shuffle=False, random_state=seed)
            clf = GridSearchCV(estimator=lasso, cv=cv, param_grid=param_grid_linear, n_jobs=n_jobs,
                               scoring=make_scorer(accuracy_score))
            clf.fit(dataset_train.data, dataset_train.target)
            best_estimator = clf.best_estimator_

            if verbose >= 3:
                print('Y_hat:', best_estimator)
                print('Relative weight for the sensible feature:', best_estimator.coef_[sensible_feature])
                print('All the weights:', best_estimator.coef_)

            # Accuracy & fairness stats
            pred = best_estimator.predict(dataset_test.data)
            pred_train = best_estimator.predict(dataset_train.data)

            acctest = accuracy_score(dataset_test.target, pred)
            acctrain = accuracy_score(dataset_train.target, pred_train)
            eqopptest = equalized_odds_measure_TP(dataset_test, best_estimator, [sensible_feature], ylabel=1)
            eqopptrain = equalized_odds_measure_TP(dataset_train, best_estimator, [sensible_feature], ylabel=1)
            nsf = len([coef for coef in best_estimator.coef_ if coef > near_zero])
            if verbose >= 2:
                print('Accuracy train:', acctrain)
                print('Accuracy test:', acctest)
                # Fairness measure
                print('Eq. opp. train: \n', eqopptrain)
                print('Eq. opp. test: \n', eqopptest)
                # Sparsity
                print('#Selected Features:', nsf)

            accuracy_train['svmV'].append(acctrain)
            accuracy_test['svmV'].append(acctest)
            eq_opp_train['svmV'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] - list(eqopptrain[sensible_feature].values())[1]))
            eq_opp_test['svmV'].append(np.abs(list(eqopptest[sensible_feature].values())[0] - list(eqopptest[sensible_feature].values())[1]))
            peq_opp_train['svmV'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] * list(eqopptrain[sensible_feature].values())[1]))
            peq_opp_test['svmV'].append(np.abs(list(eqopptest[sensible_feature].values())[0] * list(eqopptest[sensible_feature].values())[1]))
            number_selected_features['svmV'].append(nsf)

            # Train an SVM using the training set
            print('\nGrid search for the standard Linear SVM...')
            lasso = LassoC(max_iter=max_iterations)
            score, best_estimator = two_step_validation_with_DEO(dataset_train, dataset_test, lasso, verbose=verbose, n_jobs=n_jobs,
                                                                 sensible_feature=sensible_feature, params=param_grid_linear)

            if verbose >= 3:
                print('Y_hat:', best_estimator)
                print('Relative weight for the sensible feature:', best_estimator.coef_[sensible_feature])
                print('All the weights:', best_estimator.coef_)

            # Accuracy & fairness stats
            pred = best_estimator.predict(dataset_test.data)
            pred_train = best_estimator.predict(dataset_train.data)

            acctest = accuracy_score(dataset_test.target, pred)
            acctrain = accuracy_score(dataset_train.target, pred_train)
            eqopptest = equalized_odds_measure_TP(dataset_test, best_estimator, [sensible_feature], ylabel=1)
            eqopptrain = equalized_odds_measure_TP(dataset_train, best_estimator, [sensible_feature], ylabel=1)
            nsf = len([coef for coef in best_estimator.coef_ if coef > near_zero])
            if verbose >= 2:
                print('Accuracy train:', acctrain)
                print('Accuracy test:', acctest)
                # Fairness measure
                print('Eq. opp. train: \n', eqopptrain)
                print('Eq. opp. test: \n', eqopptest)
                # Sparsity
                print('#Selected Features:', nsf)

            accuracy_train['svm'].append(acctrain)
            accuracy_test['svm'].append(acctest)
            eq_opp_train['svm'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] - list(eqopptrain[sensible_feature].values())[1]))
            eq_opp_test['svm'].append(np.abs(list(eqopptest[sensible_feature].values())[0] - list(eqopptest[sensible_feature].values())[1]))
            peq_opp_train['svm'].append(np.abs(list(eqopptrain[sensible_feature].values())[0] * list(eqopptrain[sensible_feature].values())[1]))
            peq_opp_test['svm'].append(np.abs(list(eqopptest[sensible_feature].values())[0] * list(eqopptest[sensible_feature].values())[1]))
            number_selected_features['svm'].append(nsf)

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
            number_selected_features['hardt'].append(nsf)

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
            lasso = LassoC(max_iter=max_iterations)
            algorithm = UncorrelationMethod(dataset_train, model=None, sensible_feature=sensible_feature)
            new_dataset_train = algorithm.new_representation(dataset_train.data)
            new_dataset_train = namedtuple('_', 'data, target')(new_dataset_train, dataset_train.target)
            new_dataset_test = algorithm.new_representation(dataset_test.data)
            new_dataset_test = namedtuple('_', 'data, target')(new_dataset_test, dataset_test.target)
            score, best_estimator = two_step_validation_with_DEO(new_dataset_train, new_dataset_test, lasso, verbose=verbose, n_jobs=n_jobs,
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
            nsf = len([coef for coef in best_estimator.coef_ if coef > near_zero])
            if verbose >= 2:
                print('Eq. opp. train fair: \n', eqopptrain)
                print('Eq. opp. test fair: \n', eqopptest)
                # Sparsity
                print('#Selected Features:', nsf)

            accuracy_train['our'].append(facctrain)
            accuracy_test['our'].append(facctest)
            eq_opp_train['our'].append(np.abs(list(eqopptrain[0].values())[0] - list(eqopptrain[0].values())[1]))
            eq_opp_test['our'].append(np.abs(list(eqopptest[0].values())[0] - list(eqopptest[0].values())[1]))
            peq_opp_train['our'].append(np.abs(list(eqopptrain[0].values())[0] * list(eqopptrain[0].values())[1]))
            peq_opp_test['our'].append(np.abs(list(eqopptest[0].values())[0] * list(eqopptest[0].values())[1]))
            number_selected_features['our'].append(nsf)

        if verbose >= 1 and iteration != number_of_iterations - 1:
            print('\n\nStats at iteration', iteration + 1)
            print('Method \t Accuracy on train \t Accuracy on test \t Diff.Eq.Opp.train \t Diff.Eq.Opp.test'
                  '\t Prod.Eq.Opp.train \t Prod.Eq.Opp.test \t Number of feats')
            print('LassoV \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                  % (np.mean(accuracy_train['svmV']), np.std(accuracy_train['svmV']),
                     np.mean(accuracy_test['svmV']), np.std(accuracy_test['svmV']),
                     np.mean(eq_opp_train['svmV']), np.std(eq_opp_train['svmV']),
                     np.mean(eq_opp_test['svmV']), np.std(eq_opp_test['svmV']),
                     np.mean(peq_opp_train['svmV']), np.std(peq_opp_train['svmV']),
                     np.mean(peq_opp_test['svmV']), np.std(peq_opp_test['svmV']),
                     np.mean(number_selected_features['svmV']), np.std(number_selected_features['svmV'])))
            print('Lasso \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                  % (np.mean(accuracy_train['svm']), np.std(accuracy_train['svm']),
                     np.mean(accuracy_test['svm']), np.std(accuracy_test['svm']),
                     np.mean(eq_opp_train['svm']), np.std(eq_opp_train['svm']),
                     np.mean(eq_opp_test['svm']), np.std(eq_opp_test['svm']),
                     np.mean(peq_opp_train['svm']), np.std(peq_opp_train['svm']),
                     np.mean(peq_opp_test['svm']), np.std(peq_opp_test['svm']),
                     np.mean(number_selected_features['svm']), np.std(number_selected_features['svm'])))
            print('Hardt \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                  % (np.mean(accuracy_train['hardt']), np.std(accuracy_train['hardt']),
                     np.mean(accuracy_test['hardt']), np.std(accuracy_test['hardt']),
                     np.mean(eq_opp_train['hardt']), np.std(eq_opp_train['hardt']),
                     np.mean(eq_opp_test['hardt']), np.std(eq_opp_test['hardt']),
                     np.mean(peq_opp_train['hardt']), np.std(peq_opp_train['hardt']),
                     np.mean(peq_opp_test['hardt']), np.std(peq_opp_test['hardt']),
                     np.mean(number_selected_features['hardt']), np.std(number_selected_features['hardt'])))
            print('Our \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
                  % (np.mean(accuracy_train['our']), np.std(accuracy_train['our']),
                     np.mean(accuracy_test['our']), np.std(accuracy_test['our']),
                     np.mean(eq_opp_train['our']), np.std(eq_opp_train['our']),
                     np.mean(eq_opp_test['our']), np.std(eq_opp_test['our']),
                     np.mean(peq_opp_train['our']), np.std(peq_opp_train['our']),
                     np.mean(peq_opp_test['our']), np.std(peq_opp_test['our']),
                     np.mean(number_selected_features['our']), np.std(number_selected_features['our'])))

    print('\n\n\n\nFinal stats (after', iteration+1, 'iterations)')
    print('Method \t Accuracy on train \t Accuracy on test \t Diff.Eq.Opp.train \t Diff.Eq.Opp.test'
          '\t Prod.Eq.Opp.train \t Prod.Eq.Opp.test \t Number of feats')
    print(
        'LassoV \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
        % (np.mean(accuracy_train['svmV']), np.std(accuracy_train['svmV']),
           np.mean(accuracy_test['svmV']), np.std(accuracy_test['svmV']),
           np.mean(eq_opp_train['svmV']), np.std(eq_opp_train['svmV']),
           np.mean(eq_opp_test['svmV']), np.std(eq_opp_test['svmV']),
           np.mean(peq_opp_train['svmV']), np.std(peq_opp_train['svmV']),
           np.mean(peq_opp_test['svmV']), np.std(peq_opp_test['svmV']),
           np.mean(number_selected_features['svmV']), np.std(number_selected_features['svmV'])))
    print(
        'Lasso \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
        % (np.mean(accuracy_train['svm']), np.std(accuracy_train['svm']),
           np.mean(accuracy_test['svm']), np.std(accuracy_test['svm']),
           np.mean(eq_opp_train['svm']), np.std(eq_opp_train['svm']),
           np.mean(eq_opp_test['svm']), np.std(eq_opp_test['svm']),
           np.mean(peq_opp_train['svm']), np.std(peq_opp_train['svm']),
           np.mean(peq_opp_test['svm']), np.std(peq_opp_test['svm']),
           np.mean(number_selected_features['svm']), np.std(number_selected_features['svm'])))
    print(
        'Hardt \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
        % (np.mean(accuracy_train['hardt']), np.std(accuracy_train['hardt']),
           np.mean(accuracy_test['hardt']), np.std(accuracy_test['hardt']),
           np.mean(eq_opp_train['hardt']), np.std(eq_opp_train['hardt']),
           np.mean(eq_opp_test['hardt']), np.std(eq_opp_test['hardt']),
           np.mean(peq_opp_train['hardt']), np.std(peq_opp_train['hardt']),
           np.mean(peq_opp_test['hardt']), np.std(peq_opp_test['hardt']),
           np.mean(number_selected_features['hardt']), np.std(number_selected_features['hardt'])))
    print(
        'Our \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
        % (np.mean(accuracy_train['our']), np.std(accuracy_train['our']),
           np.mean(accuracy_test['our']), np.std(accuracy_test['our']),
           np.mean(eq_opp_train['our']), np.std(eq_opp_train['our']),
           np.mean(eq_opp_test['our']), np.std(eq_opp_test['our']),
           np.mean(peq_opp_train['our']), np.std(peq_opp_train['our']),
           np.mean(peq_opp_test['our']), np.std(peq_opp_test['our']),
           np.mean(number_selected_features['our']), np.std(number_selected_features['our'])))
