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


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_.ravel() #[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    inter = clf.intercept_.ravel()
    if inter == 0.0:
        yy = a * xx
    else:
        yy = a * xx - (inter / w[1])
    plt.plot(xx, yy, linestyle, label=label)

# Plots and stats
training_output = True
test_output = True

# Number of samples per component
n_samples = 100
n_samples_low = 20

lasso_dataset = True
number_of_random_features = 1000
lasso_algorithm = True

# Generate random sample, two components
np.random.seed(0)

varA = 0.8
aveApos = [-1.0, -1.0]
aveAneg = [1.0, 1.0]

varB = 0.5
aveBpos = [0.5, -0.5]
aveBneg = [0.5, 0.5]

# TRAIN DATASET
X = np.random.multivariate_normal(aveApos, [[varA, 0], [0, varA]], n_samples)
X = np.vstack([X, np.random.multivariate_normal(aveAneg, [[varA, 0], [0, varA]], n_samples)])
X = np.vstack([X, np.random.multivariate_normal(aveBpos, [[varB, 0], [0, varB]], n_samples_low)])
X = np.vstack([X, np.random.multivariate_normal(aveBneg, [[varB, 0], [0, varB]], n_samples)])
# Random features not related to the task
if lasso_dataset:
    Xrand = np.random.uniform(low=-1.0, high=1.0, size=(n_samples * 3 + n_samples_low, number_of_random_features))
    X = np.hstack([X, Xrand])
# Sensitive feature
idx_A = list(range(0, n_samples * 2))
idx_B = list(range(n_samples * 2, n_samples * 3 + n_samples_low))
sensible_feature = [1] * (n_samples * 2) + [0] * (n_samples + n_samples_low)
sensible_feature = np.array(sensible_feature)
sensible_feature.shape = (len(sensible_feature), 1)
X = np.hstack([X, sensible_feature])
sensible_feature_id = len(X[1, :]) - 1
# Labels
y = [1] * n_samples + [-1] * n_samples + [1] * n_samples_low + [-1] * n_samples
y = np.array(y)

# TEST DATASET
X_test = np.random.multivariate_normal(aveApos, [[varA, 0], [0, varA]], n_samples)
X_test = np.vstack([X_test, np.random.multivariate_normal(aveAneg, [[varA, 0], [0, varA]], n_samples)])
X_test = np.vstack([X_test, np.random.multivariate_normal(aveBpos, [[varB, 0], [0, varB]], n_samples_low)])
X_test = np.vstack([X_test, np.random.multivariate_normal(aveBneg, [[varB, 0], [0, varB]], n_samples)])
# Random features not related to the task
if lasso_dataset:
    Xrand = np.random.uniform(low=-1.0, high=1.0, size=(n_samples * 3 + n_samples_low, number_of_random_features))
    X_test = np.hstack([X_test, Xrand])
# Sensitive feature
X_test = np.hstack([X_test, sensible_feature])
# Labels
y_test = [1] * n_samples + [-1] * n_samples + [1] * n_samples_low + [-1] * n_samples
y_test = np.array(y)

cv = KFold(n_splits=10, shuffle=False, random_state=1)
if lasso_algorithm:
    basesvc = Lasso(alpha=1.0, fit_intercept=True)
    param_grid_linear = [{'alpha': np.logspace(-2, 2, 10)}]
else:
    basesvc = svm.LinearSVC(C=10.0, fit_intercept=True)#, class_weight="balanced")
    param_grid_linear = [{'C': np.logspace(-2, 2, 10)}]
svcGS = GridSearchCV(estimator=basesvc, cv=cv, param_grid=param_grid_linear, n_jobs=2)
svcGS.fit(X, y)
svc = svcGS.best_estimator_

if training_output:
    plt.figure(1)
    plt.title('Training examples')
    plt.scatter(X[:n_samples * 2, 0], X[:n_samples * 2, 1], marker='o', s=25, c=y[:n_samples * 2], edgecolors='k', label='Group A')
    plt.scatter(X[n_samples * 2:, 0], X[n_samples * 2:, 1], marker='s', s=25, c=y[n_samples * 2:], edgecolors='k', label='Group B')
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])
    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])
    plot_hyperplane(svc, min_x, max_x, 'k-.', 'General Boundary')
    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    prediction = svc.predict(X)
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('\nTrain Accuracy both groups:', accuracy_score(y, prediction))
    subgropus_idxs = subgrups_sensible_feature_data(X, sensible_feature_id)
    tpr_pred = fair_tpr_from_precomputed(y, prediction, subgropus_idxs)
    print('with EO:', tpr_pred)
    print('Coeff:', svc.coef_.ravel())
if test_output:
    plt.figure(2)
    plt.title('Test examples')
    plt.scatter(X_test[:n_samples * 2, 0], X_test[:n_samples * 2, 1], marker='o', s=25, c=y_test[:n_samples * 2], edgecolors='k', label='Group A')
    plt.scatter(X_test[n_samples * 2:, 0], X_test[n_samples * 2:, 1], marker='s', s=25, c=y_test[n_samples * 2:], edgecolors='k', label='Group B')
    min_x = np.min(X_test[:, 0])
    max_x = np.max(X_test[:, 0])
    min_y = np.min(X_test[:, 1])
    max_y = np.max(X_test[:, 1])
    plot_hyperplane(svc, min_x, max_x, 'k-.', 'General Boundary')
    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    prediction = svc.predict(X_test)
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('\nTest Accuracy both groups:', accuracy_score(y_test, prediction))
    subgropus_idxs = subgrups_sensible_feature_data(X_test, sensible_feature_id)
    tpr_pred = fair_tpr_from_precomputed(y_test, prediction, subgropus_idxs)
    print('with EO:', tpr_pred)

svcGS.fit(X[idx_A, :], y[idx_A])
svc = svcGS.best_estimator_
if training_output:
    plt.figure(1)
    plot_hyperplane(svc, min_x, max_x, 'r-.', 'Group A Boundary')
    prediction = svc.predict(X)
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('\nTrain Accuracy group A for all the examples:', accuracy_score(y, prediction))
    subgropus_idxs = subgrups_sensible_feature_data(X, sensible_feature_id)
    tpr_pred = fair_tpr_from_precomputed(y, prediction, subgropus_idxs)
    print('with EO:', tpr_pred)
    prediction = svc.predict(X[idx_A, :])
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('Train Accuracy group A on group A:', accuracy_score(y[idx_A], prediction))
    prediction = svc.predict(X[idx_B, :])
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('Train Accuracy group A on group B:', accuracy_score(y[idx_B], prediction))
if test_output:
    plt.figure(2)
    plot_hyperplane(svc, min_x, max_x, 'r-.', 'Group A Boundary')
    prediction = svc.predict(X_test)
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('\nTest Accuracy group A for all the examples:', accuracy_score(y_test, prediction))
    subgropus_idxs = subgrups_sensible_feature_data(X_test, sensible_feature_id)
    tpr_pred = fair_tpr_from_precomputed(y_test, prediction, subgropus_idxs)
    print('with EO:', tpr_pred)
    prediction = svc.predict(X_test[idx_A, :])
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('Test Accuracy group A on group A:', accuracy_score(y_test[idx_A], prediction))
    prediction = svc.predict(X_test[idx_B, :])
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('Test Accuracy group A on group B:', accuracy_score(y_test[idx_B], prediction))


svcGS.fit(X[idx_B, :], y[idx_B])
svc = svcGS.best_estimator_
if training_output:
    plt.figure(1)
    plot_hyperplane(svc, min_x, max_x, 'b-.', 'Group B Boundary')
    prediction = svc.predict(X)
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('\nTrain Accuracy group B for all the examples:', accuracy_score(y, prediction))
    subgropus_idxs = subgrups_sensible_feature_data(X, sensible_feature_id)
    tpr_pred = fair_tpr_from_precomputed(y, prediction, subgropus_idxs)
    print('with EO:', tpr_pred)
    prediction = svc.predict(X[idx_A, :])
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('Train Accuracy group B on group A:', accuracy_score(y[idx_A], prediction))
    prediction = svc.predict(X[idx_B, :])
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('Train Accuracy group B on group B:', accuracy_score(y[idx_B], prediction))
if test_output:
    plt.figure(2)
    plot_hyperplane(svc, min_x, max_x, 'b-.', 'Group B Boundary')
    prediction = svc.predict(X_test)
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('\nTest Accuracy group B for all the examples:', accuracy_score(y_test, prediction))
    subgropus_idxs = subgrups_sensible_feature_data(X_test, sensible_feature_id)
    tpr_pred = fair_tpr_from_precomputed(y_test, prediction, subgropus_idxs)
    print('with EO:', tpr_pred)
    prediction = svc.predict(X_test[idx_A, :])
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('Test Accuracy group B on group A:', accuracy_score(y_test[idx_A], prediction))
    prediction = svc.predict(X_test[idx_B, :])
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('Test Accuracy group B on group B:', accuracy_score(y_test[idx_B], prediction))

print('\n')
dataset = namedtuple('_', 'data, target')(X, y)
algorithm = UncorrelationMethod(dataset, svc, sensible_feature_id)
algorithm.fit()
if training_output:
    plt.figure(1)
    plot_hyperplane(algorithm, min_x, max_x, 'g-.', 'Fair Boundary')
    prediction = algorithm.predict(X)
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('\nFair Train Accuracy group A for all the examples:', accuracy_score(y, prediction))
    subgropus_idxs = subgrups_sensible_feature_data(X, sensible_feature_id)
    tpr_pred = fair_tpr_from_precomputed(y, prediction, subgropus_idxs)
    print('with EO:', tpr_pred)
    print('Coeff:', svc.coef_.ravel())

if test_output:
    plt.figure(2)
    plot_hyperplane(algorithm, min_x, max_x, 'g-.', 'Fair Boundary')
    prediction = algorithm.predict(X_test)
    prediction = np.sign(np.sign(prediction) + 0.1)
    print('\nFair Test Accuracy group A for all the examples:', accuracy_score(y_test, prediction))
    subgropus_idxs = subgrups_sensible_feature_data(X_test, sensible_feature_id)
    tpr_pred = fair_tpr_from_precomputed(y_test, prediction, subgropus_idxs)
    print('with EO:', tpr_pred)


plt.legend()
plt.colorbar()
plt.show()
