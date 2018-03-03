import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from measures import fair_tpr_from_precomputed, subgrups_sensible_feature_data
from uncorrelation import UncorrelationMethod
from collections import namedtuple


point_size = 100
linewidth = 5

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    if clf.intercept_ == 0.0:
        yy = a * xx
    else:
        yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, linewidth=linewidth, label=label)

# Number of samples per component
# n_samples = int(100 * 25.0 / 8.0) + 2
# n_samples_low = int(20 * 25.0 / 8.0)
n_samples = 100
n_samples_low = 20

# Generate random sample, two components
np.random.seed(0)

varA = 0.5
aveApos = [-1.0, -1.0]
aveAneg = [1.0, 1.0]

varB = 0.5
aveBpos = [0.5, -0.5]
aveBneg = [0.5, 0.5]


X = np.random.multivariate_normal(aveApos, [[varA, 0], [0, varA]], n_samples)
X = np.vstack([X, np.random.multivariate_normal(aveAneg, [[varA, 0], [0, varA]], n_samples)])
X = np.vstack([X, np.random.multivariate_normal(aveBpos, [[varB, 0], [0, varB]], n_samples_low)])
X = np.vstack([X, np.random.multivariate_normal(aveBneg, [[varB, 0], [0, varB]], n_samples)])

idx_A = list(range(0, n_samples * 2))
idx_B = list(range(n_samples * 2, n_samples * 3 + n_samples_low))

sensible_feature = [1] * (n_samples * 2) + [0] * (n_samples + n_samples_low)
sensible_feature = np.array(sensible_feature)
sensible_feature.shape = (len(sensible_feature), 1)
X = np.hstack([X, sensible_feature])
sensible_feature_id = len(X[1, :]) - 1

y = [1] * n_samples + [0] * n_samples + [1] * n_samples_low + [0] * n_samples
y = np.array(y)


np.savetxt('X_toy', X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt('Y_toy', y, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')

plt.scatter(X[:n_samples * 2, 0], X[:n_samples * 2, 1], marker='o', s=point_size, c=y[:n_samples * 2], edgecolors='k', label='Group A')
plt.scatter(X[n_samples * 2:, 0], X[n_samples * 2:, 1], marker='s', s=point_size, c=y[n_samples * 2:], edgecolors='k', label='Group B')


#svc = svm.SVC(C=10.0, kernel='linear', class_weight="balanced")
svc = svm.LinearSVC(C=10.0, fit_intercept=True)#, class_weight="balanced")
svc.fit(X, y)
min_x = np.min(X[:, 0])
max_x = np.max(X[:, 0])
min_y = np.min(X[:, 1])
max_y = np.max(X[:, 1])
plot_hyperplane(svc, min_x, max_x, 'k', 'SVM Boundary')
plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
prediction = svc.predict(X)
print('Train Accuracy both groups:', accuracy_score(y, prediction))
subgropus_idxs = subgrups_sensible_feature_data(X, sensible_feature_id)
tpr_pred = fair_tpr_from_precomputed(y, prediction, subgropus_idxs)
print('with EO:', tpr_pred)

svc.fit(X[idx_A, :], y[idx_A])
plot_hyperplane(svc, min_x, max_x, 'g--', 'Group A Boundary')
prediction = svc.predict(X)
print('Train Accuracy group A for all the examples:', accuracy_score(y, prediction))
subgropus_idxs = subgrups_sensible_feature_data(X, sensible_feature_id)
tpr_pred = fair_tpr_from_precomputed(y, prediction, subgropus_idxs)
print('with EO:', tpr_pred)

prediction = svc.predict(X[idx_A, :])
print('Train Accuracy group A on group A:', accuracy_score(y[idx_A], prediction))
prediction = svc.predict(X[idx_B, :])
print('Train Accuracy group A on group B:', accuracy_score(y[idx_B], prediction))

svc.fit(X[idx_B, :], y[idx_B])
plot_hyperplane(svc, min_x, max_x, 'b--', 'Group B Boundary')
prediction = svc.predict(X)
print('Train Accuracy group B for all the examples:', accuracy_score(y, prediction))
subgropus_idxs = subgrups_sensible_feature_data(X, sensible_feature_id)
tpr_pred = fair_tpr_from_precomputed(y, prediction, subgropus_idxs)
print('with EO:', tpr_pred)

prediction = svc.predict(X[idx_A, :])
print('Train Accuracy group B on group A:', accuracy_score(y[idx_A], prediction))
prediction = svc.predict(X[idx_B, :])
print('Train Accuracy group B on group B:', accuracy_score(y[idx_B], prediction))

print('\n')
dataset = namedtuple('_', 'data, target')(X, y)
algorithm = UncorrelationMethod(dataset, svc, sensible_feature_id)
algorithm.fit()
plot_hyperplane(algorithm, min_x, max_x, 'r', 'Fair Boundary')
prediction = algorithm.predict(X)
print('Fair Train Accuracy group A for all the examples:', accuracy_score(y, prediction))
subgropus_idxs = subgrups_sensible_feature_data(X, sensible_feature_id)
tpr_pred = fair_tpr_from_precomputed(y, prediction, subgropus_idxs)
print('with EO:', tpr_pred)

plt.legend()
# plt.colorbar()
plt.show()