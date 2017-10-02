from load_data import load_heart_uci
from measures import equalized_odds_measure_TP, equalized_odds_measure_FP, \
    statistical_parity_measure, disparate_impact_measure

from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


set_of_exps = 2

if set_of_exps == 1:
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
    print(equalized_odds_measure_TP(dataset, clf, [sens_feat], ylabel=1))
    print('Eq. Odds for Y = -1')
    print(equalized_odds_measure_TP(dataset, clf, [sens_feat], ylabel=-1))
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

elif set_of_exps == 2:
    from load_data import load_binary_diabetes_uci
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA

    diabetes = load_binary_diabetes_uci()
    # 50% for train
    ntrain = len(diabetes.target) // 2

    # Train an SVM using the training set
    clf = svm.SVC(C=10.0)
    # with C = 10 and C = 100 => same accuracy, different fairness!
    # with C = 1000 => from 76% to 74% in accuracy but fair!
    clf.fit(diabetes.data[:ntrain, :], diabetes.target[:ntrain])

    # The dataset becomes the test set
    diabetes.data = diabetes.data[ntrain:, :]
    diabetes.target = diabetes.target[ntrain:]

    # Accuracy
    pred = clf.predict(diabetes.data)
    print('Accuracy:', accuracy_score(diabetes.target, pred))

    # Fairness measure
    sens_feat = 1  # Feature 1 is the gender of the patient
    print('Eq. Odds for Y = 1')
    print(equalized_odds_measure_TP(diabetes, clf, [sens_feat], ylabel=1))
    print(equalized_odds_measure_FP(diabetes, clf, [sens_feat], ylabel=1))
    print('Eq. Odds for Y = -1')
    print(equalized_odds_measure_TP(diabetes, clf, [sens_feat], ylabel=-1))
    print(equalized_odds_measure_FP(diabetes, clf, [sens_feat], ylabel=-1))
    print('Statistical Parity for Y = 1')
    print(statistical_parity_measure(diabetes, clf, [sens_feat], ylabel=1))
    print('Statistical Parity for Y = -1')
    print(statistical_parity_measure(diabetes, clf, [sens_feat], ylabel=-1))
    print('Disparate Impact')
    print(disparate_impact_measure(diabetes, clf, [sens_feat]))

    # Plot of the FN (PCA)
    #    list_of_errors = [idx for idx in range(len(predictions)) if predictions[idx] != diabetes.target[idx]]
    #    list_of_oks = [idx for idx in range(len(predictions)) if predictions[idx] == diabetes.target[idx]]
    list_of_fn = [idx for idx in range(len(pred)) if pred[idx] != diabetes.target[idx] == 1]
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(diabetes.data)
    #    ax.scatter(X_reduced[list_of_errors, 0], X_reduced[list_of_errors, 1], X_reduced[list_of_errors, 2],
    #               c=diabetes.data[list_of_errors, 1], cmap=plt.cm.Set1, edgecolor='k', s=80)
    #    ax.scatter(X_reduced[list_of_oks, 0], X_reduced[list_of_oks, 1], X_reduced[list_of_oks, 2],
    #               c=diabetes.data[list_of_oks, 1], cmap=plt.cm.Set1, edgecolor='k', s=10)
    ax.scatter(X_reduced[list_of_fn, 0], X_reduced[list_of_fn, 1], X_reduced[list_of_fn, 2],
               c=diabetes.data[list_of_fn, 1], cmap=plt.cm.Set1, edgecolor='k', s=50)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()