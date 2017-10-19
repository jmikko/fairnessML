from load_data import load_binary_diabetes_uci, load_heart_uci, load_breast_cancer, load_adult, load_adult_race
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from measures import equalized_odds_measure_TP, equalized_odds_measure_FP, equalized_odds_measure_from_pred_TP, equalized_odds_measure_TP_from_list_of_sensfeat
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.optimize import linprog
from hardt import gamma_y_hat, HardtMethod
from scipy.spatial import ConvexHull
from uncorrelation import UncorrelationMethod
from uncorrelation_nonlinear import Fair_SVM, polynomial_kernel, gaussian_kernel, linear_kernel

# Experimental settings
experiment_number = 1
verbose = 3

number_of_iterations = 100

grid_search_complete = 1
if grid_search_complete:
    param_grid_linear = [
        {'C': [0.1, 0.5, 1, 10, 100, 1000, 10000, 100000], 'kernel': ['linear']}
    ]
    param_grid_all = [
        {'C': [0.1, 0.5, 1, 10, 100, 1000, 10000, 100000], 'kernel': ['linear']},
        {'C': [0.1, 0.5, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']},
    ]
else:
    param_grid_linear = [{'C': [10.0], 'kernel': ['linear']}]
    param_grid_all = [{'C': [10.0], 'kernel': ['rbf'], 'gamma': [0.4]}]
# ***************************************************

accuracy_train = {'hardt': [], 'hardtK': [], 'our': [], 'svm': [], 'svmK': [], 'ourK': []}
accuracy_test = {'hardt': [], 'hardtK': [], 'our': [], 'svm': [], 'svmK': [], 'ourK': []}
eq_opp_train = {'hardt': [], 'hardtK': [], 'our': [], 'svm': [], 'svmK': [], 'ourK': []}
eq_opp_test = {'hardt': [], 'hardtK': [], 'our': [], 'svm': [], 'svmK': [], 'ourK': []}


for iteration in range(number_of_iterations):
    seed = iteration
    np.random.seed(seed)
    print('\n\n\nIteration -', iteration+1)
    if experiment_number == 0:
        print('Loading diabetes dataset...')
        dataset_train = load_binary_diabetes_uci()
        dataset_test = load_binary_diabetes_uci()
        sensible_feature = 1  # sex
        if verbose == 1:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 1:
        print('Loading heart dataset...')
        dataset_train = load_heart_uci()
        dataset_test = load_heart_uci()
        sensible_feature = 1  # sex
        if verbose == 1:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 2:
        print('Loading adult (gender) dataset...')
        dataset_train, dataset_test = load_adult(smaller=False)
        sensible_feature = 9  # sex
        if verbose == 1:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 3:
        print('Loading adult (white vs. other races) dataset...')
        dataset_train, dataset_test = load_adult_race(smaller=False)
        sensible_feature = 8  # race
        if verbose == 1:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))

    if experiment_number in [0, 1]:
        # % for train
        ntrain = 8 * len(dataset_train.target) // 10
        permutation = list(range(len(dataset_train.target)))
        np.random.shuffle(permutation)
        train_idx = permutation[:ntrain]
        test_idx = permutation[ntrain:]
        dataset_train.data = dataset_train.data[train_idx, :]
        dataset_train.target = dataset_train.target[train_idx]
        dataset_test.data = dataset_test.data[test_idx, :]
        dataset_test.target = dataset_test.target[test_idx]
    if experiment_number in [2, 3]:
        ntrain = len(dataset_test.target)

    # Train an SVM using the training set
    print('\nGrid search for the standard Linear SVM...')
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid_linear, n_jobs=3)
    clf.fit(dataset_train.data, dataset_train.target)
    if verbose >= 3:
        print('Y_hat:', clf.best_estimator_)
    # Accuracy & fairness stats
    pred = clf.predict(dataset_test.data)
    pred_train = clf.predict(dataset_train.data)

    acctest = accuracy_score(dataset_test.target, pred)
    acctrain = accuracy_score(dataset_train.target, pred_train)
    eqopptest = equalized_odds_measure_TP(dataset_test, clf, [sensible_feature], ylabel=1)
    eqopptrain = equalized_odds_measure_TP(dataset_train, clf, [sensible_feature], ylabel=1)
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

    # Hardt method
    print('\nHardt method on linear SVM...')
    algorithm = HardtMethod(dataset_train, clf, sensible_feature)
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


    # Our uncorrelation method - Linear
    print('\nOur uncorrelation method...')
    list_of_sensible_feature_test = dataset_test.data[:, sensible_feature]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid_linear, n_jobs=1)
    algorithm = UncorrelationMethod(dataset_train, clf, sensible_feature)
    algorithm.fit()
    if verbose >= 3:
        print('Our Y:', algorithm.model.best_estimator_)

    # Accuracy
    pred = algorithm.predict(dataset_test.data)
    pred_train = algorithm.predict(dataset_train.data)
    facctrain = accuracy_score(dataset_train.target, pred_train)
    facctest = accuracy_score(dataset_test.target, pred)
    if verbose >= 2:
        print('Our Accuracy train:', facctest)
        print('Our Accuracy test:', facctrain)
    # Fairness measure
    eqopptrain = equalized_odds_measure_TP_from_list_of_sensfeat(dataset_train, algorithm, [algorithm.list_of_sensible_feature_train], ylabel=1)
    eqopptest = equalized_odds_measure_TP_from_list_of_sensfeat(dataset_test, algorithm, [list_of_sensible_feature_test], ylabel=1)
    if verbose >= 2:
        print('Eq. opp. train fair: \n', eqopptrain)
        print('Eq. opp. test fair: \n', eqopptest)

    accuracy_train['our'].append(facctrain)
    accuracy_test['our'].append(facctest)
    eq_opp_train['our'].append(np.abs(list(eqopptrain[0].values())[0] - list(eqopptrain[0].values())[1]))
    eq_opp_test['our'].append(np.abs(list(eqopptest[0].values())[0] - list(eqopptest[0].values())[1]))



    # Train an SVM using the training set
    print('\nGrid search for the standard Kernel SVM...')
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid_all, n_jobs=3)
    clf.fit(dataset_train.data, dataset_train.target)
    if verbose >= 3:
        print('Y_hat:', clf.best_estimator_)
    # Accuracy & fairness stats
    pred = clf.predict(dataset_test.data)
    pred_train = clf.predict(dataset_train.data)

    acctest = accuracy_score(dataset_test.target, pred)
    acctrain = accuracy_score(dataset_train.target, pred_train)
    eqopptest = equalized_odds_measure_TP(dataset_test, clf, [sensible_feature], ylabel=1)
    eqopptrain = equalized_odds_measure_TP(dataset_train, clf, [sensible_feature], ylabel=1)
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


    # Hardt method
    print('\nHardtK method...')
    algorithm = HardtMethod(dataset_train, clf, sensible_feature)
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



    # Our uncorrelation method - Kernel
    print('\nOur uncorrelation method with kernels...')
    list_of_sensible_feature_test = dataset_test.data[:, sensible_feature]
    list_of_sensible_feature_train = dataset_train.data[:, sensible_feature]

    algorithm = Fair_SVM(sensible_feature=sensible_feature)
    clf = GridSearchCV(algorithm, param_grid_all, n_jobs=3)
    clf.fit(dataset_train.data, dataset_train.target)
    if verbose >= 3:
        print('Our Y:', clf.best_estimator_)

    # Accuracy
    facctest = clf.score(dataset_test.data, dataset_test.target)
    facctrain = clf.score(dataset_train.data, dataset_train.target)
    if verbose >= 2:
        print('Our Accuracy train:', facctest)
        print('Our Accuracy test:', facctrain)
    # Fairness measure
    eqopptrain = equalized_odds_measure_TP_from_list_of_sensfeat(dataset_train, clf, [list_of_sensible_feature_train], ylabel=1)
    eqopptest = equalized_odds_measure_TP_from_list_of_sensfeat(dataset_test, clf, [list_of_sensible_feature_test], ylabel=1)
    if verbose >= 2:
        print('Eq. opp. train fair: \n', eqopptrain)
        print('Eq. opp. test fair: \n', eqopptest)

    accuracy_train['ourK'].append(facctrain)
    accuracy_test['ourK'].append(facctest)
    eq_opp_train['ourK'].append(np.abs(list(eqopptrain[0].values())[0] - list(eqopptrain[0].values())[1]))
    eq_opp_test['ourK'].append(np.abs(list(eqopptest[0].values())[0] - list(eqopptest[0].values())[1]))

    if verbose >= 1 and iteration != number_of_iterations - 1:
        print('Stats at iteration', iteration)
        print('Method \t Accuracy on train \t Accuracy on test \t Diff.Eq.Opp.train \t Diff.Eq.Opp.test')
        print('SVM \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['svm']), np.std(accuracy_train['svm']),
                 np.mean(accuracy_test['svm']), np.std(accuracy_test['svm']),
                 np.mean(eq_opp_train['svm']), np.std(eq_opp_train['svm']),
                 np.mean(eq_opp_test['svm']), np.std(eq_opp_test['svm'])))
        print('Hardt \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['hardt']), np.std(accuracy_train['hardt']),
                 np.mean(accuracy_test['hardt']), np.std(accuracy_test['hardt']),
                 np.mean(eq_opp_train['hardt']), np.std(eq_opp_train['hardt']),
                 np.mean(eq_opp_test['hardt']), np.std(eq_opp_test['hardt'])))
        print('Our \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['our']), np.std(accuracy_train['our']),
                 np.mean(accuracy_test['our']), np.std(accuracy_test['our']),
                 np.mean(eq_opp_train['our']), np.std(eq_opp_train['our']),
                 np.mean(eq_opp_test['our']), np.std(eq_opp_test['our'])))
        print('SVMK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['svmK']), np.std(accuracy_train['svmK']),
                 np.mean(accuracy_test['svmK']), np.std(accuracy_test['svmK']),
                 np.mean(eq_opp_train['svmK']), np.std(eq_opp_train['svmK']),
                 np.mean(eq_opp_test['svmK']), np.std(eq_opp_test['svmK'])))
        print('HardtK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['hardtK']), np.std(accuracy_train['hardtK']),
                 np.mean(accuracy_test['hardtK']), np.std(accuracy_test['hardtK']),
                 np.mean(eq_opp_train['hardtK']), np.std(eq_opp_train['hardtK']),
                 np.mean(eq_opp_test['hardtK']), np.std(eq_opp_test['hardtK'])))
        print('OurK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
              % (np.mean(accuracy_train['ourK']), np.std(accuracy_train['ourK']),
                 np.mean(accuracy_test['ourK']), np.std(accuracy_test['ourK']),
                 np.mean(eq_opp_train['ourK']), np.std(eq_opp_train['ourK']),
                 np.mean(eq_opp_test['ourK']), np.std(eq_opp_test['ourK'])))

print('\n\n\n\nFinal stats (after', iteration+1, 'iterations)')
print('Method \t Accuracy on train \t Accuracy on test \t Diff.Eq.Opp.train \t Diff.Eq.Opp.test')
print('SVM \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
      % (np.mean(accuracy_train['svm']), np.std(accuracy_train['svm']),
         np.mean(accuracy_test['svm']), np.std(accuracy_test['svm']),
         np.mean(eq_opp_train['svm']), np.std(eq_opp_train['svm']),
         np.mean(eq_opp_test['svm']), np.std(eq_opp_test['svm'])))
print('Hardt \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
      % (np.mean(accuracy_train['hardt']), np.std(accuracy_train['hardt']),
         np.mean(accuracy_test['hardt']), np.std(accuracy_test['hardt']),
         np.mean(eq_opp_train['hardt']), np.std(eq_opp_train['hardt']),
         np.mean(eq_opp_test['hardt']), np.std(eq_opp_test['hardt'])))
print('Our \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
      % (np.mean(accuracy_train['our']), np.std(accuracy_train['our']),
         np.mean(accuracy_test['our']), np.std(accuracy_test['our']),
         np.mean(eq_opp_train['our']), np.std(eq_opp_train['our']),
         np.mean(eq_opp_test['our']), np.std(eq_opp_test['our'])))
print('SVMK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
      % (np.mean(accuracy_train['svmK']), np.std(accuracy_train['svmK']),
         np.mean(accuracy_test['svmK']), np.std(accuracy_test['svmK']),
         np.mean(eq_opp_train['svmK']), np.std(eq_opp_train['svmK']),
         np.mean(eq_opp_test['svmK']), np.std(eq_opp_test['svmK'])))
print('HardtK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
      % (np.mean(accuracy_train['hardtK']), np.std(accuracy_train['hardtK']),
         np.mean(accuracy_test['hardtK']), np.std(accuracy_test['hardtK']),
         np.mean(eq_opp_train['hardtK']), np.std(eq_opp_train['hardtK']),
         np.mean(eq_opp_test['hardtK']), np.std(eq_opp_test['hardtK'])))
print('OurK \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f \t %.3f +- %.3f'
      % (np.mean(accuracy_train['ourK']), np.std(accuracy_train['ourK']),
         np.mean(accuracy_test['ourK']), np.std(accuracy_test['ourK']),
         np.mean(eq_opp_train['ourK']), np.std(eq_opp_train['ourK']),
         np.mean(eq_opp_test['ourK']), np.std(eq_opp_test['ourK'])))

