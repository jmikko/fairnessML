import numpy as np
import sklearn.datasets
import pandas as pd
import sklearn.preprocessing as preprocessing
from collections import namedtuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


def load_heart_uci():
    '''
    Features:
    0. age
    1. sex
    2. cp
    3. trestbps
    4. chol
    5. fbs
    6. restecg
    7. thalach
    8. exang
    9. oldpeak
    10. slope
    11. ca
    12. thal
    '''
    dataset = sklearn.datasets.fetch_mldata('heart')
    return dataset


def load_binary_diabetes_uci():
    '''
    Features:
    0. Age
    1. Sex
    2. Body mass index
    3. Average blood pressure
    4-9. S1-S6
    '''
    dataset = sklearn.datasets.load_diabetes()
    # Make the target binary: high progression Vs. low progression of the disease
    dataset.target = np.array([1 if diabetes_progression > 139 else -1 for diabetes_progression in dataset.target])
    val0 = np.min(dataset.data[0, 1])
    dataset.data[:, 1] = [0 if val == val0 else 1 for val in dataset.data[:, 1]]
    return dataset


def load_breast_cancer():
    dataset = sklearn.datasets.load_breast_cancer()
    # dataset.target = [y for y in dataset.target]
    return dataset


def load_adult(smaller=False, scaler=True):
    '''
    Features:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    data = pd.read_csv(
        "./datasets/adult/adult.data",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"],
            # dtype=object,
            # sep=r'\s*,\s*',
            # engine='python',
            #na_values="?"
            )
    len_train = len(data.as_matrix()[:, -1])
    data_test = pd.read_csv(
        "./datasets/adult/adult.test",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"],
            # dtype=object,
            # sep=r'\s*,\s*',
            # engine='python',
            #na_values="?"
            )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    # len_all = len(data.as_matrix()[:, -1])
    domanda = data["workclass"][4].values[1]
    # print(domanda)
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # len_clean = len(data.as_matrix()[:, -1])
    # len_diff = len_all - len_clean
    # print(len_train, len_all, len_clean)
    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.as_matrix()
    target = np.array([-1.0 if val == 0 else 1.0 for val in np.array(datamat)[:, -1]])
    # print(data["income"])
    datamat = datamat[:, :-1]
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if smaller:
        data = namedtuple('_', 'data, target')(datamat[:len_train // 5, :], target[:len_train // 5])
        data_test = namedtuple('_', 'data, target')(datamat[len_train:, :], target[len_train:])
    else:
        data = namedtuple('_', 'data, target')(datamat[:len_train, :-1], target[:len_train])
        data_test = namedtuple('_', 'data, target')(datamat[len_train:, :-1], target[len_train:])
    return data, data_test


def load_adult_race(A1=['white'], smaller=False, scaler=True):
    # Feature 8 is "race"
    # race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    data_train, data_test = load_adult(smaller, scaler)
    A1_val = []
    race_white_value = data_train.data[0][8]
    race_black_value = data_train.data[3][8]
    race_asian_value = data_train.data[11][8]
    race_amer_value = data_train.data[15][8]
    race_other_value = data_train.data[50][8]
    for strings in A1:
        val = None
        if strings == 'white':
            val = race_white_value
        elif strings == 'black':
            val = race_black_value
        elif strings == 'asian':
            val = race_asian_value
        elif strings == 'amer-indian':
            val = race_amer_value
        elif strings == 'other':
            val = race_other_value
        else:
            print('Error in A1 argument - string not found!')
            return 0, 0
        A1_val.append(val)

    for idx in range(len(data_train.data)):
        data_train.data[idx][8] = 1.0 if data_train.data[idx][8] in A1_val else -1.0
    for idx in range(len(data_test.data)):
        data_test.data[idx][8] = 1.0 if data_test.data[idx][8] in A1_val else -1.0
    return data_train, data_test

if __name__ == "__main__":

    from sklearn import svm
    #  data = sklearn.datasets.fetch_mldata('iris')
    data, data_test = load_adult_race(smaller=False)

    print('Train examples #', len(data.target), 'pos | neg :', len([0.0 for val in data.target if val == 1]), '|', len([0.0 for val in data.target if val == -1]))
    print('Test examples #', len(data_test.target), 'pos | neg :', len([0.0 for val in data_test.target if val == 1]), '|', len([0.0 for val in data_test.target if val == -1]))

    print(data.data[0, :], data.target[0])
    print(data.data[1, :], data.target[1])
    print(data.data[2, :], data.target[2])

    for i in range(len(data.data[1, :])):
        print(i, '# =', len(set(data.data[:, i])))

    from sklearn.metrics import accuracy_score
    svc = svm.SVC(C=0.5, class_weight="balanced")
    svc.fit(data.data, data.target)
    print('Data train #ex #negative ex:', len(data.target), np.count_nonzero(data.target + 1))
    prediction = svc.predict(data.data)
    print('Train #ex and #negative ex:', len(prediction), np.count_nonzero(prediction + 1))
    numn = len([1.0 for y in data.target if y == -1])
    nump = len(data.target) - numn
    print('Train Accuracy Balanced:', accuracy_score(data.target, prediction,
                                                     sample_weight=[1.0 / numn if y == -1 else 1.0 / nump for y in data.target]))
    print('Train Accuracy:', accuracy_score(data.target, prediction))

    prediction = svc.predict(data_test.data)
    print('Data test #ex #negative ex:', len(data_test.target), np.count_nonzero(data_test.target + 1))
    prediction = svc.predict(data_test.data)
    print('Test #ex and #negative ex:', len(prediction), np.count_nonzero(prediction + 1))
    numn = len([1.0 for y in data_test.target if y == -1])
    nump = len(data_test.target) - numn
    #  print(nn, np)
    print('Test Accuracy Balanced:', accuracy_score(data_test.target, prediction,
                                           sample_weight=[1.0 / numn if y == -1 else 1.0 / nump for y in data_test.target]))
    print('Test Accuracy:', accuracy_score(data_test.target, prediction))

