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


def load_adult_race_white_vs_black(smaller=False, scaler=True, balanced=False):
    # Feature 8 is "race"
    # race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    data_train, data_test = load_adult(smaller, scaler)
    race_white_value = data_train.data[0][8]
    race_black_value = data_train.data[3][8]
    race_asian_value = data_train.data[11][8]
    race_amer_value = data_train.data[15][8]
    race_other_value = data_train.data[50][8]

    A1_val = race_white_value
    A0_val = race_black_value

    new_train_data = np.array([el for el in data_train.data if el[8] in [A1_val, A0_val]])
    new_train_target = np.array([el for idx, el in enumerate(data_train.target) if data_train.data[idx][8] in [A1_val, A0_val]])
    new_test_data = np.array([el for el in data_test.data if el[8] in [A1_val, A0_val]])
    new_test_target = np.array([el for idx, el in enumerate(data_test.target) if data_test.data[idx][8] in [A1_val, A0_val]])

    if balanced:
        idx_white = [idx for idx, el in enumerate(new_train_data) if el[8] == A1_val]
        idx_black = [idx for idx, el in enumerate(new_train_data) if el[8] == A0_val]

        min_group = np.min([len(idx_white), len(idx_black)])

        idx_white = idx_white[:min_group]
        idx_black = idx_black[:min_group]
        idxs = idx_white + idx_black

        data_train = namedtuple('_', 'data, target')(new_train_data[idxs], new_train_target[idxs])
        data_test = namedtuple('_', 'data, target')(new_test_data, new_test_target)
    else:
        data_train = namedtuple('_', 'data, target')(new_train_data, new_train_target)
        data_test = namedtuple('_', 'data, target')(new_test_data, new_test_target)

    for idx in range(len(data_train.data)):
        data_train.data[idx][8] = 1.0 if data_train.data[idx][8] == A1_val else -1.0
    for idx in range(len(data_test.data)):
        data_test.data[idx][8] = 1.0 if data_test.data[idx][8] == A1_val else -1.0
    return data_train, data_test

def laod_propublica_fairml_hotencoded():
    """ Features:
    0. Two_yr_Recidivism
    1. Number_of_Priors
    2. Age_Above_FourtyFive
    3. Age_Below_TwentyFive
    4. African_American
    5. Asian
    6. Hispanic
    7. Native_American
    9. Other
    10. Female
    11. Misdemeanor

    Target: score_factor
    """
    # read in propublica data
    propublica_data = pd.read_csv("./"
                                  "propublica_data_for_fairml.csv")
    # quick data processing
    compas_rating = propublica_data.score_factor.values
    propublica_data = propublica_data.drop("score_factor", 1)

    newFemale = [val if val == 1.0 else -1.0 for val in propublica_data.Female.values]
    propublica_data = propublica_data.drop("Female", 1)
    propublica_data.insert(10, 'Female', newFemale)

    dataset = namedtuple('_', 'data, target')(np.array(propublica_data.values), np.array(compas_rating))
    return dataset

def laod_propublica_fairml():
    """ Features:
    0. Two_yr_Recidivism
    1. Number_of_Priors
    2. Age_Above_FourtyFive
    3. Age_Below_TwentyFive
    4. Female
    5. Misdemeanor
    6. Race

    Target: score_factor
    """
    dataset = laod_propublica_fairml_hotencoded()
    # read in propublica data
    propublica_data = pd.read_csv("./"
                                  "propublica_data_for_fairml.csv")
    # quick data processing
    compas_rating = propublica_data.score_factor.values
    propublica_data = propublica_data.drop("score_factor", 1)

    black_race_list = propublica_data.African_American.values * 1
    asian_race_list = propublica_data.Asian.values * 2
    hispanic_race_list = propublica_data.Hispanic.values * 3
    native_race_list = propublica_data.Native_American.values * 4
    other_race_list = propublica_data.Other.values * 5

    feature_race_list = black_race_list + asian_race_list + hispanic_race_list + native_race_list + other_race_list

    propublica_data = propublica_data.drop("African_American", 1)
    propublica_data = propublica_data.drop("Asian", 1)
    propublica_data = propublica_data.drop("Hispanic", 1)
    propublica_data = propublica_data.drop("Native_American", 1)
    propublica_data = propublica_data.drop("Other", 1)

    propublica_data.insert(6, 'Race', feature_race_list)

    dataset = namedtuple('_', 'data, target')(np.array(propublica_data.values), np.array(compas_rating))
    return dataset

def laod_propublica_fairml_race(A1=[1]):
    '''
    Values of the feature number 6:
        black_race = 1
        asian_race = 2
        hispanic_race = 3
        native_race = 4
        other_race = 5
    '''

    dataset = laod_propublica_fairml()
    for idx in range(len(dataset.data)):
        dataset.data[idx][6] = 1.0 if dataset.data[idx][6] in A1 else -1.0
    return dataset

def load_default(remove_categorical=False, smaller=False, scaler=True):
    '''
        0. X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
        1. X2: Gender (1 = male; 2 = female).
        2. X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
        3. X4: Marital status (1 = married; 2 = single; 3 = others).
        4. X5: Age (year).
        5 - 10. X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
        11 - 16. X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
        17 - 22. X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.
        target: Y = default payment next month (+1 or -1)
    '''
    dataset = pd.read_excel("./default_credit_card_clients.xls")
    # dataset = dataset.drop("ID", 1)
    default_payment = dataset.Y.values
    dataset = dataset.drop("Y", 1)

    default_payment = default_payment[1:]
    default_payment = np.array([el if el == 1.0 else -1.0 for el in default_payment])

    if remove_categorical:
        dataset = dataset.drop("X3", 1)
        dataset = dataset.drop("X4", 1)

    if scaler:
        scaler = StandardScaler()
        scaler.fit(np.array(dataset.values[1:, :], dtype=np.float))
        dataset = scaler.transform(dataset.values[1:, :])

    if smaller:
        all_idxs = list(range(len(default_payment)))
        np.random.shuffle(all_idxs)
        selected_idxs = all_idxs[:10000]
        dataset = namedtuple('_', 'data, target')(dataset[selected_idxs, :], default_payment[selected_idxs])
    else:
        dataset = namedtuple('_', 'data, target')(dataset, default_payment)
    return dataset

def load_hepatitis():
    from scipy.stats import mode
    hepatitis = pd.read_csv("./datasets/hepatitis/data.txt", header=-1)
    hepatitis = hepatitis.as_matrix()
    hepatitis = np.where(np.isnan(hepatitis), mode(hepatitis, axis=0), hepatitis)
    y = hepatitis[:, -1]
    x = hepatitis[:, :-1]
    dataset = namedtuple('_', 'data, target')(x, y)
    return dataset

if __name__ == "__main__":
    load_hepatitis()
    load_default()
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

