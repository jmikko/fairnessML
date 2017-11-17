import numpy as np
from sklearn.metrics import confusion_matrix


def subgrups_sensible_feature(data, sensible_feature):
    dict_idxs = {}
    values_of_sensible_feature = list(set(data.data[:, sensible_feature]))
    for val in values_of_sensible_feature:
        dict_idxs[val] = [idx for idx, x in enumerate(data.data) if x[sensible_feature] == val]
    return dict_idxs


def fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fp) / float(fp + tn)


def tpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tp) / float(tp + fn)


def fair_tpr_from_model(data, model, sensible_feature):
    predictions = model.predict(data.data)
    truth = data.target
    dict_idxs = subgrups_sensible_feature(data, sensible_feature)
    for val in dict_idxs:
        dict_idxs[val] = tpr(truth[dict_idxs[val]], predictions[dict_idxs[val]])
    return dict_idxs


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


def equalized_odds_measure_TP_from_list_of_sensfeat(data, model, sensible_features, ylabel, rev_pred=1):
    predictions = model.predict(data.data) * rev_pred
    truth = data.target
    eq_dict = {}
    for idf, features in enumerate(sensible_features):
        eq_sensible_feature = {}
        values_of_sensible_feature = list(set(features))
        for val in values_of_sensible_feature:
            eq_tmp = None
            positive_sensitive = np.sum([1.0 if features[i] == val and truth[i] == ylabel else 0.0
                                         for i in range(len(predictions))])
            if positive_sensitive > 0:
                eq_tmp = np.sum([1.0 if predictions[i] == ylabel and features[i] == val and truth[i] == ylabel
                                 else 0.0 for i in range(len(predictions))]) / positive_sensitive
            eq_sensible_feature[val] = eq_tmp
        eq_dict[idf] = eq_sensible_feature
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


# TODO: check this
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

    # New measures
    idxs = subgrups_sensible_feature(diabetes, sensible_feature=1)
    print('TPR')
    print(fair_tpr_from_model(diabetes, clf, sensible_feature=1))


