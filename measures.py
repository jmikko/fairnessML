import numpy as np


def equalized_odds_measure_TP(data, model, sensible_features, ylabel, rev_pred=1):
    predictions = model.predict(data.data) * rev_pred
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
                eq_tmp = np.sum([1.0 if predictions[i] == ylabel and data.data[i, feature] == val and truth[i] == ylabel
                                 else 0.0 for i in range(len(predictions))]) / positive_sensitive
            eq_sensible_feature[val] = eq_tmp
        eq_dict[feature] = eq_sensible_feature
    return eq_dict


def equalized_odds_measure_from_pred_TP(data, pred, sensible_features, ylabel, rev_pred=1):
    predictions = pred * rev_pred
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
                eq_tmp = np.sum([1.0 if predictions[i] == ylabel and data.data[i, feature] == val and truth[i] == ylabel
                                 else 0.0 for i in range(len(predictions))]) / positive_sensitive
            eq_sensible_feature[val] = eq_tmp
        eq_dict[feature] = eq_sensible_feature
    return eq_dict


def equalized_odds_measure_FP(data, model, sensible_features, ylabel, rev_pred=1):
    predictions = model.predict(data.data) * rev_pred
    truth = data.target
    eq_dict = {}
    for feature in sensible_features:
        eq_sensible_feature = {}
        values_of_sensible_feature = list(set(data.data[:, feature]))
        for val in values_of_sensible_feature:
            eq_tmp = None
            positive_sensitive = np.sum([1.0 if data.data[i, feature] == val and truth[i] != ylabel else 0.0
                                         for i in range(len(predictions))])
            if positive_sensitive > 0:
                eq_tmp = np.sum([1.0 if predictions[i] == ylabel and data.data[i, feature] == val and truth[i] != ylabel
                                 else 0.0 for i in range(len(predictions))]) / positive_sensitive
            eq_sensible_feature[val] = eq_tmp
        eq_dict[feature] = eq_sensible_feature
    return eq_dict


# TODO: questo non so se sia giusto!!
def false_omission_rate(data, model, sensible_features, ylabel):
    predictions = model.predict(data.data)
    truth = data.target
    eq_dict = {}
    for feature in sensible_features:
        eq_sensible_feature = {}
        values_of_sensible_feature = list(set(data.data[:, feature]))
        for val in values_of_sensible_feature:
            eq_tmp = None
            positive_sensitive = np.sum([1.0 if data.data[i, feature] == val and truth[i] != ylabel else 0.0
                                         for i in range(len(predictions))])
            if positive_sensitive > 0:
                eq_tmp = np.sum([1.0 if predictions[i] == ylabel and data.data[i, feature] == val and truth[i] != ylabel
                                 else 0.0 for i in range(len(predictions))]) / positive_sensitive
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

    diabetes = load_binary_diabetes_uci()
    # 50% for train
    ntrain = len(diabetes.target) // 2

    # Train an SVM using the training set
    clf = svm.SVC(C=10.0)
    clf.fit(diabetes.data[:ntrain, :], diabetes.target[:ntrain])

    # The dataset becomes the test set
    diabetes.data = diabetes.data[ntrain:, :]
    diabetes.target = diabetes.target[ntrain:]

    # Accuracy
    pred = clf.predict(diabetes.data)
    print('Accuracy:',  accuracy_score(diabetes.target, pred))

    # Fairness measure
    print(equalized_odds_measure_TP(diabetes, clf, [1], ylabel=1))  # Feature 1 is SEX


