import numpy as np
import sklearn.datasets


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
    dataset.target = [1 if diabetes_progression > 139 else -1 for diabetes_progression in dataset.target]
    return dataset

if __name__ == "__main__":
    data = load_heart_uci()
    print(data.data[1, :], data.target[1])
