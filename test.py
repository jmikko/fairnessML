from load_data import load_heart_uci
from measures import equalized_odds_measure, statistical_parity_measure, disparate_impact_measure

from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

dataset = load_heart_uci()
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset.data = scaler.fit(dataset.data).transform(dataset.data)
# print(heart.data[1, :], heart.data[2, :], heart.data[3, :])
# % for train
ntrain = 5 * len(dataset.target) // 10  # from 30% to 50% the accuracy is the same BUT more fairness!

# Train an SVM using the training set
clf = svm.SVC(C=1.0)
clf.fit(dataset.data[:ntrain, :], dataset.target[:ntrain])

# The dataset becomes the test set
dataset.data = dataset.data[ntrain:, :]
dataset.target = dataset.target[ntrain:]

# Accuracy
pred = clf.predict(dataset.data)
print('Accuracy:',  accuracy_score(dataset.target, pred))

# Fairness measure
sens_feat = 1  # Feature 1 is the gender of the patient
print('Eq. Odds for Y = 1')
print(equalized_odds_measure(dataset, clf, [sens_feat], ylabel=1))
print('Eq. Odds for Y = -1')
print(equalized_odds_measure(dataset, clf, [sens_feat], ylabel=-1))
print('Statistical Parity for Y = 1')
print(statistical_parity_measure(dataset, clf, [sens_feat], ylabel=1))
print('Statistical Parity for Y = -1')
print(statistical_parity_measure(dataset, clf, [sens_feat], ylabel=-1))
print('Disparate Impact')
print(disparate_impact_measure(dataset, clf, [sens_feat]))

# Plot of the FN (PCA)
list_of_fn = [idx for idx in range(len(pred)) if pred[idx] != dataset.target[idx] and dataset.target[idx] == 1]
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(dataset.data)
ax.scatter(X_reduced[list_of_fn, 0], X_reduced[list_of_fn, 1], X_reduced[list_of_fn, 2],
           c=dataset.data[list_of_fn, sens_feat], cmap=plt.cm.Set1, edgecolor='k', s=50)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()