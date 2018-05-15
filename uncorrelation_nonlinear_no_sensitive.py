from load_data import load_binary_diabetes_uci, load_heart_uci, load_breast_cancer, load_adult, load_adult_race
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from measures import equalized_odds_measure_TP, equalized_odds_measure_FP, equalized_odds_measure_TP_from_list_of_sensfeat
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.optimize import linprog
from hardt import gamma_y_hat, HardtMethod
from scipy.spatial import ConvexHull
from collections import namedtuple
from cvxopt import solvers, matrix
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
from sklearn.base import BaseEstimator


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * (linalg.norm(x - y)**2))


class Fair_SVM_no_sensitive(BaseEstimator):
    def __init__(self, kernel='rbf', C=1.0, sensible_feature=None, gamma=1.0):
        self.kernel = kernel
        self.C = C
        self.fairness = False if sensible_feature is None else True
        self.sensible_feature = sensible_feature
        self.gamma = gamma
        self.w = None

    def fit(self, X, y):
        if self.kernel == 'rbf':
            self.fkernel = lambda x, y: gaussian_kernel(x, y, self.gamma)
        elif self.kernel == 'linear':
            self.fkernel = linear_kernel
        else:
            self.fkernel = linear_kernel

        if self.fairness:
            self.values_of_sensible_feature = list(set(X[:, self.sensible_feature]))
            self.list_of_sensible_feature_train = X[:, self.sensible_feature]
            self.val0 = np.min(self.values_of_sensible_feature)
            self.val1 = np.max(self.values_of_sensible_feature)
            self.set_A1 = [idx for idx, ex in enumerate(X) if y[idx] == 1
                           and ex[self.sensible_feature] == self.val1]
            self.set_not_A1 = [idx for idx, ex in enumerate(X) if y[idx] == 1
                               and ex[self.sensible_feature] == self.val0]
            self.set_1 = [idx for idx, ex in enumerate(X) if y[idx] == 1]
            self.n_A1 = len(self.set_A1)
            self.n_not_A1 = len(self.set_not_A1)
            self.n_1 = len(self.set_1)

            X = np.delete(X, self.sensible_feature, 1)

        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.fkernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # print(y)
        A = cvxopt.matrix(y.astype(np.double), (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # print(self.C)
        # Stack the fairness constraint
        if self.fairness:
            tau = [(np.sum(K[self.set_A1, idx]) / self.n_A1) - (np.sum(K[self.set_not_A1, idx]) / self.n_not_A1)
                   for idx in range(len(y))]
            fairness_line = matrix(y * tau, (1, n_samples), 'd')
            A = cvxopt.matrix(np.vstack([A, fairness_line]))
            b = cvxopt.matrix([0.0, 0.0])

        # solve QP problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-7
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        # print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.fkernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def decision_function(self, X):
        return self.project(X)

    def predict(self, X):
        return np.sign(self.project(X))

    def score(self, X_test, y_test):
        predict = self.predict(X_test)
        acc = accuracy_score(y_test, predict)
        # print('acc', acc)
        return acc


if __name__ == "__main__":
    experiment_number = 0
    if experiment_number == 0:
        dataset_train = load_binary_diabetes_uci()
        dataset_test = load_binary_diabetes_uci()
        sensible_feature = 1  # sex
    elif experiment_number == 1:
        dataset_train = load_heart_uci()
        dataset_test = load_heart_uci()
        sensible_feature = 1  # sex
    elif experiment_number == 2:
        dataset_train, dataset_test = load_adult(smaller=False)
        sensible_feature = 9  # sex
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 3:
        dataset_train, dataset_test = load_adult_race(smaller=False)
        sensible_feature = 8  # race
        print('Different values of the sensible feature', sensible_feature, ':',
              set(dataset_train.data[:, sensible_feature]))

    if experiment_number in [0, 1]:
        # % for train
        ntrain = 5 * len(dataset_train.target) // 10
        dataset_train.data = dataset_train.data[:ntrain, :]
        dataset_train.target = dataset_train.target[:ntrain]
        dataset_test.data = dataset_test.data[ntrain:, :]
        dataset_test.target = dataset_test.target[ntrain:]
    if experiment_number in [2, 3]:
        ntrain = len(dataset_test.target)

    # Standard SVM
    # Train an SVM using the training set
    print('Grid search for SVM...')
    grid_search_complete = 1
    if grid_search_complete:
        param_grid = [
            #{'C': [0.1, 0.5, 1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [0.1, 0.5, 1, 10, 100, 1000], 'gamma': [1.0 / len(dataset_train.data[1, :]), 0.1, 0.01, 0.001], 'kernel': ['rbf']},
        ]
    else:
        param_grid = [{'C': [10.0], 'kernel': ['linear'], 'gamma': [1.0 / len(dataset_train.data[1, :])]}]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid, n_jobs=1)
    clf.fit(dataset_train.data, dataset_train.target)
    print('Y:', clf.best_estimator_)

    # Accuracy
    pred = clf.predict(dataset_test.data)
    pred_train = clf.predict(dataset_train.data)
    print('Accuracy test:', accuracy_score(dataset_test.target, pred))
    print('Accuracy train:', accuracy_score(dataset_train.target, pred_train))
    # Fairness measure
    print('Eq. opp. test: \n',
          equalized_odds_measure_TP(dataset_test, clf, [sensible_feature], ylabel=1))
    print('Eq. opp. train: \n',
          equalized_odds_measure_TP(dataset_train, clf, [sensible_feature], ylabel=1))

    #  Our method test
    print('\n\n\nGrid search for our method...')
    algorithm = Fair_SVM(sensible_feature=sensible_feature)
    clf = GridSearchCV(algorithm, param_grid, n_jobs=1)
    clf.fit(dataset_train.data, dataset_train.target)
    print('Y:', clf.best_estimator_)

    y_predict = clf.predict(dataset_test.data)
    # correct = np.sum(y_predict == dataset_test.target)
    # print("%d out of %d predictions correct" % (correct, len(y_predict)))

    # Accuracy
    pred = clf.predict(dataset_test.data)
    pred_train = clf.predict(dataset_train.data)
    print('Accuracy test:', accuracy_score(dataset_test.target, pred))
    print('Accuracy train:', accuracy_score(dataset_train.target, pred_train))
    # Fairness measure
    print('Eq. opp. test: \n',
          equalized_odds_measure_TP(dataset_test, clf, [sensible_feature], ylabel=1))
    print('Eq. opp. train: \n',
          equalized_odds_measure_TP(dataset_train, clf, [sensible_feature], ylabel=1))