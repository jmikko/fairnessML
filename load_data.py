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
    dataset.target = np.array([1.0 if y == 1 else -1.0for y in dataset.target])
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
    compas_rating = np.array([1.0 if y > 0 else -1.0 for y in compas_rating])
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
    compas_rating = np.array([1.0 if y > 0 else -1.0 for y in compas_rating])
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
    hepatitis = np.where(np.isnan(hepatitis), mode(hepatitis, axis=0), hepatitis)[1]
    y = np.array([1.0 if yy == 1 else -1.0 for yy in hepatitis[:, -1]])
    x = hepatitis[:, :-1]
    dataset = namedtuple('_', 'data, target')(x, y)
    return dataset


def load_arrhythmia():
    from scipy.stats import mode
    arrhythmia = pd.read_csv("./datasets/arrhythmia/arrhythmia.data.txt", header=-1)
    arrhythmia = arrhythmia.as_matrix()
    arrhythmia = np.where(np.isnan(arrhythmia), mode(arrhythmia, axis=0), arrhythmia)[1]
    y = np.array([1.0 if yy == 1 else -1.0 for yy in arrhythmia[:, -1]])
    x = arrhythmia[:, :-1]
    dataset = namedtuple('_', 'data, target')(x, y)
    return dataset


def load_german():
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    g = pd.read_csv("./datasets/german/german.data.txt", header=-1, sep='\s+')
    g = g.as_matrix()
    g = np.array(g, dtype='str')
    g = LabelEncoder().fit_transform(g.ravel()).reshape(*g.shape)
    list_of_cat = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18]
    for i in range(len(g[1, :])):
        if len(set(g[:, i])) > 2:
            list_of_cat.append(i)
    val19_0 = np.min(g[:, 19])  # Foreign\not foreign feature
    val19_1 = np.max(g[:, 19])
    for idx, ex in enumerate(g):
        g[idx, 19] = -1.0 if g[idx, 19] == val19_0 else 1.0
    list_of_cat = sorted(list(set(list_of_cat)))
    enc = OneHotEncoder(n_values='auto', categorical_features=list_of_cat, sparse=False, handle_unknown='error')
    enc.fit(g)
    g = enc.transform(g)
    ytrue_value = g[0, -1]
    y = -np.array([1.0 if yy == ytrue_value else -1.0 for yy in g[:, -1]])
    x = g[:, :-1]
    dataset = namedtuple('_', 'data, target')(x, y)
    return dataset


def load_drug():
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    g = pd.read_csv("./datasets/drug/drug_consumption.data.txt", header=-1, sep=',')
    g = np.array(g.as_matrix())
    data = g[:, 1:13]  # Remove the ID and labels
    labels = g[:, 13:]
    yfalse_value = 'CL0'
    y = -np.array([-1.0 if yy == yfalse_value else 1.0 for yy in labels[:, 5]])
    dataset = namedtuple('_', 'data, target')(data, y)
    return dataset


# # # # # # # LOAD EXPERIMENTS
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8 COMPAS (black vs other races) dataset with race not hotencoded
# 9
# 10
# 11
# 12 Arrhythmia (gender) dataset for task: Normal Vs All-the-others
# 13
# 14
# 15 Arrhythmia (gender) dataset for task: Normal Vs All-the-others [-50% of training set]
# 16 Arrhythmia (gender) dataset for task: Normal Vs All-the-others [-75% of training set]
# 17 Arrhythmia (gender) dataset for task: Normal Vs All-the-others [-12.5 of training set]
def load_experiments(experiment_number, smaller_option=False, verbose=0):
    iteration = 0
    if experiment_number == 0:
        print('Loading diabetes dataset...')
        dataset_train = load_binary_diabetes_uci()
        dataset_test = load_binary_diabetes_uci()
        sensible_feature = 1  # sex
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 1:
        print('Loading heart dataset...')
        dataset_train = load_heart_uci()
        dataset_test = load_heart_uci()
        sensible_feature = 1  # sex
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 2:
        print('Loading adult (gender) dataset...')
        dataset_train, dataset_test = load_adult(smaller=smaller_option)
        sensible_feature = 9  # sex
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 3:
        print('Loading adult (white vs. other races) dataset...')
        dataset_train, dataset_test = load_adult_race(smaller=smaller_option)
        sensible_feature = 8  # race
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 4:
        print('Loading adult (gender) dataset by splitting the training data...')
        dataset_train, _ = load_adult(smaller=smaller_option)
        sensible_feature = 9  # sex
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 5:
        print('Loading adult (white vs. other races)  dataset by splitting the training data...')
        dataset_train, _ = load_adult_race(smaller=smaller_option)
        sensible_feature = 8  # race
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 6:
        print('Loading adult (white vs. black)  dataset by splitting the training data...')
        dataset_train, _ = load_adult_race_white_vs_black(smaller=smaller_option)
        sensible_feature = 8  # race
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 7:
        print('Loading propublica_fairml (gender) dataset with race not hotencoded...')
        dataset_train = laod_propublica_fairml()
        sensible_feature = 4  # gender
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 8:
        print('Loading propublica_fairml (black vs other races) dataset with race not hotencoded...')
        dataset_train = laod_propublica_fairml_race()
        sensible_feature = 5  # race
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 9:
        print('Loading propublica_fairml (gender) dataset with race hotencoded...')
        dataset_train = laod_propublica_fairml_hotencoded()
        sensible_feature = 10  # gender
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 10:
        print('Loading Default (gender) dataset [other categoricals are removed!]...')
        dataset_train = load_default(remove_categorical=True, smaller=smaller_option, scaler=True)
        sensible_feature = 1  # gender
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 11:
        print('Loading Hepatitis (gender) dataset...')
        dataset_train = load_hepatitis()
        sensible_feature = 2  # gender
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 12:
        print('Loading Arrhythmia (gender) dataset for task: Normal Vs All-the-others...')
        dataset_train = load_arrhythmia()
        sensible_feature = 1  # gender
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 13:
        print('Loading German (foreign or not) dataset...')
        dataset_train = load_german()
        sensible_feature = 19  # gender
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 14:
        print('Loading Drug (black vs others) dataset... [task 16]')
        dataset_train = load_drug()
        sensible_feature = 4  # ethnicity
        print(dataset_train.data[:, sensible_feature])
        dataset_train.data[:, sensible_feature] = [1.0 if el == -0.31685 else -1.0 for el in dataset_train.data[:, sensible_feature]]
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 15:
        print('Loading Arrhythmia (gender) dataset for task: Normal Vs All-the-others... [-25% training set]')
        dataset_train = load_arrhythmia()
        sensible_feature = 1  # gender
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 16:
        print('Loading Arrhythmia (gender) dataset for task: Normal Vs All-the-others...[-50% training set]')
        dataset_train = load_arrhythmia()
        sensible_feature = 1  # gender
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))
    elif experiment_number == 17:
        print('Loading Arrhythmia (gender) dataset for task: Normal Vs All-the-others...[-75% training set]')
        dataset_train = load_arrhythmia()
        sensible_feature = 1  # gender
        if verbose >= 1 and iteration == 0:
            print('Different values of the sensible feature', sensible_feature, ':',
                  set(dataset_train.data[:, sensible_feature]))

    if experiment_number in [0, 1]:
        # % for train
        ntrain = 9 * len(dataset_train.target) // 10
        ntest = len(dataset_train.target) - ntrain
        permutation = list(range(len(dataset_train.target)))
        np.random.shuffle(permutation)
        train_idx = permutation[:ntrain]
        test_idx = permutation[ntrain:]
        dataset_train.data = dataset_train.data[train_idx, :]
        dataset_train.target = dataset_train.target[train_idx]
        dataset_test.data = dataset_test.data[test_idx, :]
        dataset_test.target = dataset_test.target[test_idx]
    if experiment_number in [2, 3]:
        ntrain = len(dataset_train.target)
        ntest = len(dataset_test.target)
        number_of_iterations = 1
        print('Only 1 iteration: train and test already with fixed split!')
    if experiment_number in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
        # % for train
        ntrain = 9 * len(dataset_train.target) // 10
        ntest = len(dataset_train.target) - ntrain
        if experiment_number == 15:
           ntrain = int(ntrain * 0.75)
           ntest = len(dataset_train.target) - ntrain
        elif experiment_number == 16:
            ntrain = int(ntrain * 0.5)
            ntest = len(dataset_train.target) - ntrain
        elif experiment_number == 17:
            ntrain = int(ntrain * 0.25)
            ntest = len(dataset_train.target) - ntrain

        permutation = list(range(len(dataset_train.target)))
        np.random.shuffle(permutation)
        train_idx = permutation[:ntrain]
        test_idx = permutation[ntrain:]
        dataset_test = namedtuple('_', 'data, target')(dataset_train.data[test_idx, :], dataset_train.target[test_idx])
        dataset_train = namedtuple('_', 'data, target')(dataset_train.data[train_idx, :],
                                                        dataset_train.target[train_idx])


    if verbose >= 1:
        print('Training examples:', ntrain)
        print('Test examples:', ntest)
        print('Number of features:', len(dataset_train.data[1, :]))
        values_of_sensible_feature = list(set(dataset_train.data[:, sensible_feature]))
        val0 = np.min(values_of_sensible_feature)
        val1 = np.max(values_of_sensible_feature)
        print('Examples in training in the first group:',
              len([el for el in dataset_train.data if el[sensible_feature] == val1]))
        print('Label True:', len([el for idx, el in enumerate(dataset_train.data) if
                                  el[sensible_feature] == val1 and dataset_train.target[idx] == 1]))
        print('Examples in training in the second group:',
              len([el for el in dataset_train.data if el[sensible_feature] == val0]))
        print('Label True:', len([el for idx, el in enumerate(dataset_train.data) if
                                  el[sensible_feature] == val0 and dataset_train.target[idx] == 1]))
        print('Examples in test in the first group:',
              len([el for el in dataset_test.data if el[sensible_feature] == val1]))
        print('Label True:', len([el for idx, el in enumerate(dataset_test.data) if
                                  el[sensible_feature] == val1 and dataset_test.target[idx] == 1]))
        print('Examples in test in the second group:',
              len([el for el in dataset_test.data if el[sensible_feature] == val0]))
        print('Label True:', len([el for idx, el in enumerate(dataset_test.data) if
                                  el[sensible_feature] == val0 and dataset_test.target[idx] == 1]))

        return dataset_train, dataset_test, sensible_feature

if __name__ == "__main__":
    for loadf in [load_heart_uci, load_binary_diabetes_uci, load_breast_cancer, laod_propublica_fairml_hotencoded, laod_propublica_fairml,
                  laod_propublica_fairml_race, load_default, load_hepatitis, load_arrhythmia, load_german, load_drug]:
        print('Load function:', loadf)
        data = loadf()
        print('Train examples # =', len(data.target), '       pos | neg =', len([0.0 for val in data.target if val == 1]), '|',
              len([0.0 for val in data.target if val == -1]))
        print(data.data[0, :], data.target[0])
        print(data.data[1, :], data.target[1])
        print(data.data[2, :], data.target[2])
        for i in range(len(data.data[1, :])):
            print(i, '# =', len(set(data.data[:, i])))
        print('\n\n\n')

    for loadf in [load_adult, load_adult_race, load_adult_race_white_vs_black]:
        data, data_test = loadf()
        print('Train examples #', len(data.target), 'pos | neg :', len([0.0 for val in data.target if val == 1]), '|',
              len([0.0 for val in data.target if val == -1]))
        print('Test examples #', len(data_test.target), 'pos | neg :',
              len([0.0 for val in data_test.target if val == 1]), '|',
              len([0.0 for val in data_test.target if val == -1]))
        print(data.data[0, :], data.target[0])
        print(data.data[1, :], data.target[1])
        print(data.data[2, :], data.target[2])
        for i in range(len(data.data[1, :])):
            print(i, '# =', len(set(data.data[:, i])))
        print('\n\n\n')


    from sklearn import svm
    #  data = sklearn.datasets.fetch_mldata('iris')
    data, data_test = load_adult_race(smaller=False)
    #  data, data_test, sensible_feature = load_experiments(14, verbose=2)

    print('Train examples #', len(data.target), 'pos | neg :', len([0.0 for val in data.target if val == 1]), '|', len([0.0 for val in data.target if val == -1]))
    print('Test examples #', len(data_test.target), 'pos | neg :', len([0.0 for val in data_test.target if val == 1]), '|', len([0.0 for val in data_test.target if val == -1]))

    print(data.data[0, :], data.target[0])
    print(data.data[1, :], data.target[1])
    print(data.data[2, :], data.target[2])

    for i in range(len(data.data[1, :])):
        print(i, '# =', len(set(data.data[:, i])))

    from sklearn.metrics import accuracy_score
    svc = svm.SVC(C=10.0, class_weight="balanced")
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

