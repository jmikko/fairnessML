import numpy as np


def equalized_odds_measure(data, model, sensible_features):
    predictions = model.predict(data.data)
    truth = data.target
    eq_dict = {}
    for feature in sensible_features:
        eq_sensible_feature = {}
        values_of_sensible_feature = list(set(data.data[:, feature]))
        for val in values_of_sensible_feature:
            eq_tmp = None
            positive_sensitive = np.sum([1.0 if data.data[i, feature] == val and truth[i] == 1 else 0.0
                                         for i in range(len(predictions))])
            if positive_sensitive > 0:
                eq_tmp = np.sum([1.0 if predictions[i] == 1 and data.data[i, feature] == val and truth[i] == 1 else 0.0
                                 for i in range(len(predictions))]) / positive_sensitive
            eq_sensible_feature[val] = eq_tmp
        eq_dict[feature] = eq_sensible_feature
    return eq_dict


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn import svm

    diabetes = datasets.load_diabetes()
    # Make the target binary
    diabetes.target = [1 if diabetes_progression > 139 else -1 for diabetes_progression in diabetes.target]
    model = svm.SVC()
    model.fit(diabetes.data, diabetes.target)

    print(equalized_odds_measure(diabetes, model, [1]))  # Feature 1 is SEX
