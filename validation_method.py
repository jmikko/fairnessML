from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from sklearn import datasets, svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

from load_data import load_binary_diabetes_uci, load_heart_uci, load_breast_cancer,\
    load_adult, load_adult_race, load_adult_race_white_vs_black, laod_propublica_fairml, laod_propublica_fairml_race,\
    laod_propublica_fairml_hotencoded, load_default

from measures import fpr, fair_DEO_from_precomputed, subgrups_sensible_feature, subgrups_sensible_feature_data, tpr
from collections import namedtuple

from sklearn.model_selection import KFold

print('Loading heart dataset...')
dataset_train = load_heart_uci()
dataset_test = load_heart_uci()
sensible_feature = 1  # sex
print('Different values of the sensible feature', sensible_feature, ':', set(dataset_train.data[:, sensible_feature]))
ntrain = 8 * len(dataset_train.target) // 10
ntest = len(dataset_train.target) - ntrain
permutation = list(range(len(dataset_train.target)))
np.random.shuffle(permutation)
train_idx = permutation[:ntrain]
test_idx = permutation[ntrain:]
dataset_test = namedtuple('_', 'data, target')(dataset_train.data[test_idx, :], dataset_train.target[test_idx])
dataset_train = namedtuple('_', 'data, target')(dataset_train.data[train_idx, :], dataset_train.target[train_idx])

svc = svm.SVC(kernel='linear')
Cs = np.logspace(-5, 1, 10)

clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=1, scoring=make_scorer(accuracy_score))
clf.fit(dataset_train.data, dataset_train.target)

print('Best score:', clf.best_score_)
max_accuracy = clf.best_score_
min_accepted_accuracy = max_accuracy * 0.9

print(clf.cv_results_['params'])
print(clf.best_estimator_.C)
print(clf.cv_results_['mean_test_score'])

idx_accepted_accuracy = [idx for idx, val in enumerate(clf.cv_results_['mean_test_score']) if val >= min_accepted_accuracy]
print(idx_accepted_accuracy)
new_Cs = Cs[idx_accepted_accuracy]
print(new_Cs)

random_state_inner = 1
inner_validation_dict = {}
for c in new_Cs:
    inner_inner_DEO = []
    cv = KFold(n_splits=3, shuffle=False, random_state=random_state_inner)
    cv_split = cv.split(dataset_train.data)
    for inner_train, inner_test in cv_split:
        dict_idxs = subgrups_sensible_feature_data(dataset_train.data[inner_test], 1)
        inner_svc = svm.SVC(kernel='linear', C=c)
        inner_svc.fit(dataset_train.data[inner_train], dataset_train.target[inner_train])
        inner_test_prediction = inner_svc.predict(dataset_train.data[inner_test])
        inner_inner_DEO.append(fair_DEO_from_precomputed(dataset_train.target[inner_test],
                                                         inner_test_prediction, dict_idxs))
    # print('Inner Inner Deo:', inner_inner_DEO)
    inner_validation_dict[c] = np.mean(inner_inner_DEO)
    # print('Inner valid dict:', inner_validation_dict)

print('DEO validation:', inner_validation_dict)

min_value = min(inner_validation_dict.values())  # minimum value
final_best_C = [k for k, v in inner_validation_dict.items() if v == min_value][0] # getting all keys containing the `minimum`
print(min_value, final_best_C)
print('Selected C:', final_best_C, ' with DEO:', min_value)

svc = svm.SVC(kernel='linear', C=final_best_C)
svc.fit(dataset_train.data, dataset_train.target)
# Prediction performance on test set is not as good as on train set
print(svc.score(dataset_test.data, dataset_test.target))
