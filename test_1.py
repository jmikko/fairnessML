from load_data import load_binary_diabetes_uci, load_heart_uci, load_breast_cancer, load_adult, load_adult_race
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from measures import equalized_odds_measure_TP, equalized_odds_measure_FP, equalized_odds_measure_from_pred_TP
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.optimize import linprog
from hardt import gamma_y_hat, HardtMethod
from scipy.spatial import ConvexHull

seed = 1
np.random.seed(seed)

experiment_number = 0
if experiment_number == 0:
    print('Loading diabetes dataset...')
    dataset_train = load_binary_diabetes_uci()
    dataset_test = load_binary_diabetes_uci()
    sensible_feature = 1  # sex
    print('Different values of the sensible feature', sensible_feature, ':',
          set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 1:
    print('Loading heart dataset...')
    dataset_train = load_heart_uci()
    dataset_test = load_heart_uci()
    sensible_feature = 1  # sex
    print('Different values of the sensible feature', sensible_feature, ':',
          set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 2:
    print('Loading adult (gender) dataset...')
    dataset_train, dataset_test = load_adult(smaller=False)
    sensible_feature = 9  # sex
    print('Different values of the sensible feature', sensible_feature, ':',
          set(dataset_train.data[:, sensible_feature]))
elif experiment_number == 3:
    print('Loading adult (white vs. other races) dataset...')
    dataset_train, dataset_test = load_adult_race(smaller=False)
    sensible_feature = 8  # race
    print('Different values of the sensible feature', sensible_feature, ':',
          set(dataset_train.data[:, sensible_feature]))

if experiment_number in [0, 1]:
    # % for train
    ntrain = 5 * len(dataset_train.target) // 10
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
print('Grid search...')
grid_search_complete = 1
if grid_search_complete:
    param_grid = [
        {'C': [0.1, 0.5, 1, 10, 100, 1000], 'kernel': ['linear']},
        #{'C': [0.1, 0.5, 1, 10, 100, 1000], 'gamma': ['auto', 0.001, 0.0001], 'kernel': ['rbf']},
    ]
else:
    param_grid = [{'C': [10.0], 'kernel': ['rbf'], 'gamma': ['auto']}]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid, n_jobs=3)
clf.fit(dataset_train.data, dataset_train.target)
print('Y_hat:', clf.best_estimator_)

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
# gamma_a(Y_hat)
print('Gamma points')
points_pos = gamma_y_hat(dataset_test, clf, [sensible_feature], ylabel=1)[sensible_feature]
points_neg = gamma_y_hat(dataset_test, clf, [sensible_feature], ylabel=1, rev_pred=-1)[sensible_feature]
print(points_pos)
print(points_neg)
# P_a(Y_hat)
print('Convex hulls')
convex_hulls = []
for value in points_neg:
    convex_hulls.append(np.array([[0, 0], points_pos[value], points_neg[value], [1, 1]]))
print(convex_hulls)
for idx, convex_hull in enumerate(convex_hulls):
    points = convex_hull
    hull = ConvexHull(points)
    plt.plot(points[:, 0], points[:, 1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        #  plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
        plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')
    plt.plot([el[0] for el in convex_hull], [el[1] for el in convex_hull], '*')

# Algorithm
algorithm = HardtMethod(dataset_train, clf, sensible_feature)
res = algorithm.fit()

if res.status == 0:
    print('Thetas:', res.x[:4])
    print('Alphas:', res.x[4:])
else:
    print('res.x:', res.x)

# res.x = [1.0, 1.0, 1.0, 1.0]
#  res.status = 0

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
    print('Fair Accuracy test:', accuracy_score(dataset_train.target, fair_pred_train))
    print('Fair Accuracy train:', accuracy_score(dataset_test.target, fair_pred))  # sul train?
    acc_Y_hat_test = accuracy_score(dataset_test.target, pred)
    acc_Y_hat_train = accuracy_score(dataset_train.target, pred_train)
    y_tilde_equals_y_hat = theta_11 * phi_hat_11 + \
                           theta_01 * phi_hat_01 + \
                           theta_10 * phi_hat_10 + \
                           theta_00 * phi_hat_00
    print('Fair Accuracy Theoretical test:', (2 * acc_Y_hat_test - 1) * y_tilde_equals_y_hat + 1 - acc_Y_hat_test)
    print('Fair Accuracy Theoretical train:', (2 * acc_Y_hat_train - 1) * y_tilde_equals_y_hat + 1 - acc_Y_hat_train)
    # Fairness measure
    print('Eq. opp. test: \n', equalized_odds_measure_from_pred_TP(dataset_test, fair_pred, [sensible_feature],
                                                                   ylabel=1))  # Feature 1 is SEX
    print('Eq. opp. train: \n', equalized_odds_measure_from_pred_TP(dataset_train, fair_pred_train, [sensible_feature],
                                                                    ylabel=1))  # Feature 1 is SEX # sul train?
    print('Equal Opportunity constraint:')
    tmp = [1.0 if pred_train[idx] == 1 else 0.0 for idx in range(ntrain)
           if dataset_train.data[idx, sensible_feature] == val0 and dataset_train.target[idx] == 1]
    psi_hat_101 = np.sum(tmp) / len(tmp)

    tmp = [1.0 if pred_train[idx] == 1 else 0.0 for idx in range(ntrain)
           if dataset_train.data[idx, sensible_feature] == val1 and dataset_train.target[idx] == 1]
    psi_hat_111 = np.sum(tmp) / len(tmp)
    alpha = psi_hat_101
    beta = psi_hat_111
    ret = theta_11 * ((1 - 2 * beta) * beta) + \
          theta_10 * ((2 * alpha - 1) * alpha) + \
          theta_01 * ((1 - 2 * beta) * (1 - beta)) + \
          theta_00 * ((2 * alpha - 1) * (1 - alpha))
    print(ret, '=', alpha - beta)

    #  Plot of the picked Theta
    xA0 = (2 * alpha - 1) * (theta_10 * alpha + theta_00 * (1 - alpha)) + 1 - alpha
    xA1 = (2 * beta - 1) * (theta_11 * beta + theta_01 * (1 - beta)) + 1 - beta
    tmp = [1.0 if pred_train[idx] == 1 else 0.0 for idx in range(ntrain)
           if dataset_train.data[idx, sensible_feature] == val0 and dataset_train.target[idx] == -1]
    psi_hat_100 = np.sum(tmp) / len(tmp)
    tmp = [1.0 if pred_train[idx] == 1 else 0.0 for idx in range(ntrain)
           if dataset_train.data[idx, sensible_feature] == val1 and dataset_train.target[idx] == -1]
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