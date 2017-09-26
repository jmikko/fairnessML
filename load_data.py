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


if __name__ == "__main__":
    data = load_heart_uci()
    print(data.data[1, :], data.target[1])
