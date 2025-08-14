import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
np.random.seed(42)

def load_data(path):
    """ reads and returns the pandas DataFrame """
    return pd.read_csv(path)

def adjust_labels(y):
    """adjust labels of season from {0,1,2,3} to {0,1}"""
    new_arr = [1 if x > 1 else 0 for x in y] # shortened function
    return np.array(new_arr)

def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.0001^2)
    """
    noise = np.random.normal(loc=0, scale=0.0001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=42)


class StandardScaler:

    def __init__(self):
        """ object instantiation """
        self.mean = None
        self.std = None

    def fit(self, X):
        """ fit scaler by learning the mean and standard deviation per feature """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        """ transform X by learned mean and standard deviation, and return it """
        for columns in range(X.shape[1]):
            for rows in range(X.shape[0]):
                X[rows, columns] = X[rows, columns] - self.mean[columns]
                X[rows, columns] = X[rows, columns] / self.std[columns]

        return X

    def fit_transform(self, X):
        """ fit scaler by learning the mean and standard deviation per feature, and then transform X """
        self.fit(X)

        X = self.transform(X)

        return X
