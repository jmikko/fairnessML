# Algorithm from Equality of Opportunity in Supervised Learning
# by M. Hardt et all

import numpy as np
from measures import equalized_odds_measure_TP, equalized_odds_measure_FP, equalized_odds_measure_from_pred_TP
import matplotlib.pyplot as plt


def gamma_y_hat(data, model, sensible_features, ylabel, rev_pred=1):
    y = equalized_odds_measure_TP(data, model, sensible_features, ylabel, rev_pred)
    x = equalized_odds_measure_FP(data, model, sensible_features, ylabel, rev_pred)
    for feature in x:
        for value in x[feature]:
            x[feature][value] = [x[feature][value], y[feature][value]]
    return x


def y_tilde(example, A, model, theta_11, theta_01, theta_10, theta_00):
    pred = model.predict(example)
    if pred == 1:
        if A == 1:
            rand = np.random.random()
            if rand < theta_11:
                return pred
            else:
                return pred * -1
        else:
            rand = np.random.random()
            if rand < theta_10:
                return pred
            else:
                return pred * -1
    else:
        if A == 1:
            rand = np.random.random()
            if rand < theta_01:
                return pred
            else:
                return pred * -1
        else:
            rand = np.random.random()
            if rand < theta_00:
                return pred
            else:
                return pred * -1


if __name__ == "__main__":
    from load_data import load_binary_diabetes_uci, load_heart_uci
    from sklearn import svm
    from sklearn.metrics import accuracy_score

    experiment_number = 0
    if experiment_number == 0:
        dataset_train = load_binary_diabetes_uci()
        dataset_test = load_binary_diabetes_uci()
    else:
        dataset_train = load_heart_uci()
        dataset_test = load_heart_uci()
    # % for train
    ntrain = 8 * len(dataset_train.target) // 10

    # The dataset becomes the test set
    dataset_train.data = dataset_train.data[:ntrain, :]
    dataset_train.target = dataset_train.target[:ntrain]
    dataset_test.data = dataset_test.data[ntrain:, :]
    dataset_test.target = dataset_test.target[ntrain:]

    # Train an SVM using the training set
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(dataset_train.data[:ntrain, :], dataset_train.target[:ntrain])

    # Accuracy
    pred = clf.predict(dataset_test.data)
    pred_train = clf.predict(dataset_train.data[:ntrain, :])
    print('Accuracy test:', accuracy_score(dataset_test.target, pred))
    print('Accuracy train:', accuracy_score(dataset_train.target, pred_train))

    # Fairness measure
    print('Eq. opp. test: \n', equalized_odds_measure_TP(dataset_test, clf, [1], ylabel=1))  # Feature 1 is SEX
    print('Eq. opp. train: \n', equalized_odds_measure_TP(dataset_train, clf, [1], ylabel=1))  # Feature 1 is SEX

    # gamma_a(Y_hat)
    print('Gamma points')
    points_pos = gamma_y_hat(dataset_test, clf, [1], ylabel=1)[1]  # Feature 1 is SEX
    points_neg = gamma_y_hat(dataset_test, clf, [1], ylabel=1, rev_pred=-1)[1]
    print(points_pos)
    print(points_neg)

    # P_a(Y_hat)
    print('Convex hulls')
    convex_hulls = []
    for value in points_neg:
        convex_hulls.append(np.array([[0, 0], points_pos[value], points_neg[value], [1, 1]]))
    print(convex_hulls)

    for idx, convex_hull in enumerate(convex_hulls):
        from scipy.spatial import ConvexHull

        points = convex_hull
        hull = ConvexHull(points)
        plt.plot(points[:, 0], points[:, 1], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
            #  plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
            plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')
        plt.plot([el[0] for el in convex_hull], [el[1] for el in convex_hull], '*')
    # plt.show()

    # Algorithm
    # pred_train = dataset_train.target
    values_of_sensible_feature = list(set(dataset_train.data[:, 1]))  # Feature 1 is SEX
    val0 = np.min(values_of_sensible_feature)
    val1 = np.max(values_of_sensible_feature)
    tmp = [1.0 if pred_train[idx] == 1 and dataset_train.data[idx, 1] == val1 else 0.0 for idx in range(ntrain)]
    phi_hat_11 = np.sum(tmp) / len(tmp)

    tmp = [1.0 if pred_train[idx] == -1 and dataset_train.data[idx, 1] == val1 else 0.0 for idx in range(ntrain)]
    phi_hat_01 = np.sum(tmp) / len(tmp)

    tmp = [1.0 if pred_train[idx] == 1 and dataset_train.data[idx, 1] == val0 else 0.0 for idx in range(ntrain)]
    phi_hat_10 = np.sum(tmp) / len(tmp)

    tmp = [1.0 if pred_train[idx] == -1 and dataset_train.data[idx, 1] == val0 else 0.0 for idx in range(ntrain)]
    phi_hat_00 = np.sum(tmp) / len(tmp)
    #  print(phi_hat_00 + phi_hat_10 + phi_hat_01 + phi_hat_11)

    #    tmp = [1.0 if pred_train[idx] == -1 else 0.0 for idx in range(ntrain)
    #           if dataset_train.data[idx, 1] == val0 and dataset_train.target[idx] == 1]
    #    psi_hat_001 = np.sum(tmp) / len(tmp)

    #    tmp = [1.0 if pred_train[idx] == -1 else 0.0 for idx in range(ntrain)
    #           if dataset_train.data[idx, 1] == val1 and dataset_train.target[idx] == 1]
    #    psi_hat_011 = np.sum(tmp) / len(tmp)

    tmp = [1.0 if pred_train[idx] == 1 else 0.0 for idx in range(ntrain)
           if dataset_train.data[idx, 1] == val0 and dataset_train.target[idx] == 1]
    psi_hat_101 = np.sum(tmp) / len(tmp)

    tmp = [1.0 if pred_train[idx] == 1 else 0.0 for idx in range(ntrain)
           if dataset_train.data[idx, 1] == val1 and dataset_train.target[idx] == 1]
    psi_hat_111 = np.sum(tmp) / len(tmp)
    #  print(psi_hat_001 + psi_hat_101, psi_hat_011 + psi_hat_111)

    from scipy.optimize import linprog

    #  Hypothesis = Y_hat is better than the random

    #  Minimize: c ^ T * x
    #  Subject
    #  to: A_ub * x <= b_ub
    #  A_eq * x == b_eq

    # x = theta_11, theta_01, theta_10, theta_00
    hat_errors = 1.0 - accuracy_score(dataset_train.target, pred_train)
    c = (2 * hat_errors - 1) * np.array([phi_hat_11, phi_hat_01, phi_hat_10, phi_hat_00])
    A_ub = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                     [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    b_ub = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    A_eq = np.array([[(1 - 2 * psi_hat_111) * psi_hat_111,
                      (1 - 2 * psi_hat_111) * (1 - psi_hat_111),
                      (2 * psi_hat_101 - 1) * psi_hat_101,
                      (2 * psi_hat_101 - 1) * (1 - psi_hat_101)]])
    b_eq = np.array([psi_hat_101 - psi_hat_111])

    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=None, method='simplex', callback=None, options=None)
    print('Thetas:', res.x)

    #  res.x = [1.0, 1.0, 1.0, 1.0]
    #  res.status = 0

    if res.status != 0:
        print('res.status != 0:')
    else:
        theta_11, theta_01, theta_10, theta_00 = res.x

        fair_pred_train = [float(y_tilde(ex.reshape(1, -1), 1 if ex[1] == val1 else 0,
                                         clf, theta_11, theta_01, theta_10, theta_00)) for ex in dataset_train.data]
        fair_pred = [float(y_tilde(ex.reshape(1, -1), 1 if ex[1] == val1 else 0,
                                   clf, theta_11, theta_01, theta_10, theta_00)) for ex in dataset_test.data]

        # Accuracy
        print('Fair Accuracy test:', accuracy_score(dataset_train.target, fair_pred_train))
        print('Fair Accuracy train:', accuracy_score(dataset_test.target, fair_pred))  # sul train?
        # Fairness measure
        print('Eq. opp. test: \n', equalized_odds_measure_from_pred_TP(dataset_test, fair_pred, [1],
                                                                       ylabel=1))  # Feature 1 is SEX
        print('Eq. opp. train: \n', equalized_odds_measure_from_pred_TP(dataset_train, fair_pred_train, [1],
                                                                        ylabel=1))  # Feature 1 is SEX # sul train?
        print('Equal Opportunity constraint:')
        alpha = psi_hat_101
        beta = psi_hat_111
        res = theta_11 * ((1 - 2 * beta) * beta) + \
              theta_10 * ((2 * alpha - 1) * alpha) + \
              theta_01 * ((1 - 2 * beta) * (1 - beta)) + \
              theta_00 * ((2 * alpha - 1) * (1 - alpha))
        print(res, '=', alpha - beta)

        #  Plot of the picked Theta
        xA0 = (2 * alpha - 1) * (theta_10 * alpha + theta_00 * (1 - alpha)) + 1 - alpha
        xA1 = (2 * beta - 1) * (theta_11 * beta + theta_01 * (1 - beta)) + 1 - beta

        tmp = [1.0 if pred_train[idx] == 1 else 0.0 for idx in range(ntrain)
               if dataset_train.data[idx, 1] == val0 and dataset_train.target[idx] == -1]
        psi_hat_100 = np.sum(tmp) / len(tmp)
        tmp = [1.0 if pred_train[idx] == 1 else 0.0 for idx in range(ntrain)
               if dataset_train.data[idx, 1] == val1 and dataset_train.target[idx] == -1]
        psi_hat_110 = np.sum(tmp) / len(tmp)

        alphai = psi_hat_100
        betai = psi_hat_110
        yA0 = (2 * alphai - 1) * (theta_10 * alphai + theta_00 * (1 - alphai)) + 1 - alphai
        yA1 = (2 * betai - 1) * (theta_11 * betai + theta_01 * (1 - betai)) + 1 - betai
        # plt.plot([x], [y], 'ro')
        plt.axhline(xA0, color='b')
        plt.axhline(xA1, color='r')
        plt.axvline(yA0, color='b')
        plt.axvline(yA1, color='r')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()
