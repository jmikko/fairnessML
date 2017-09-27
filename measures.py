import numpy as np


def equalized_odds_measure(data, model, sensible_features, ylabel):
    predictions = model.predict(data.data)
    truth = data.target
    eq_dict = {}
    for feature in sensible_features:
        eq_sensible_feature = {}
        values_of_sensible_feature = list(set(data.data[:, feature]))
        for val in values_of_sensible_feature:
            eq_tmp = None
            positive_sensitive = np.sum([1.0 if data.data[i, feature] == val and truth[i] == ylabel else 0.0
                                         for i in range(len(predictions))])
            if positive_sensitive > 0:
                eq_tmp = np.sum([1.0 if predictions[i] == 1 and data.data[i, feature] == val and truth[i] == 1 else 0.0
                                 for i in range(len(predictions))]) / positive_sensitive
            eq_sensible_feature[val] = eq_tmp
        eq_dict[feature] = eq_sensible_feature
    return eq_dict


def statistical_parity_measure(data, model, sensible_features, ylabel):
    predictions = model.predict(data.data)
    sp_dict = {}
    for feature in sensible_features:
        sp_sensible_feature = {}
        values_of_sensible_feature = list(set(data.data[:, feature]))
        for val in values_of_sensible_feature:
            sp_tmp = None
            n_sensitive = np.sum([1.0 if data.data[i, feature] == val else 0.0 for i in range(len(predictions))])
            if n_sensitive > 0:
                sp_tmp = np.sum([1.0 if predictions[i] == ylabel and data.data[i, feature] == val else 0.0
                                 for i in range(len(predictions))]) / n_sensitive
            sp_sensible_feature[val] = sp_tmp
        sp_dict[feature] = sp_sensible_feature
    return sp_dict


def disparate_impact_measure(data, model, sensible_features):
    di_dict = statistical_parity_measure(data, model, sensible_features, ylabel=1)
    for feature in di_dict:
        values_of_sensible_feature = list(set(data.data[:, feature]))
        if len(values_of_sensible_feature) != 2:
            di_dict[feature] = {}
        else:
            di_dict[feature] = np.min([di_dict[feature][values_of_sensible_feature[0]] /
                                       di_dict[feature][values_of_sensible_feature[1]],
                                       di_dict[feature][values_of_sensible_feature[1]] /
                                       di_dict[feature][values_of_sensible_feature[0]]])
    return di_dict


if __name__ == "__main__":
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
    clf = svm.SVC(C=1000.0)
    # with C = 10 and C = 100 => same accuracy, different fairness!
    # with C = 1000 => from 76% to 74% in accuracy but fair!
    clf.fit(diabetes.data[:ntrain, :], diabetes.target[:ntrain])

    # The dataset becomes the test set
    diabetes.data = diabetes.data[ntrain:, :]
    diabetes.target = diabetes.target[ntrain:]

    # Accuracy
    pred = clf.predict(diabetes.data)
    print('Accuracy:',  accuracy_score(diabetes.target, pred))

    # Fairness measure
    print(equalized_odds_measure(diabetes, clf, [1]))  # Feature 1 is SEX

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
