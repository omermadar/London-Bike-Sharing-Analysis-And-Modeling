import numpy as np
from statistics import mode
from abc import abstractmethod, ABC
from data import StandardScaler




class KNN(ABC):
    def __init__(self, k):
        """ object instantiation, save k and define a scaler object """
        self.k = k
        self.x_train = None
        self.y_train = None
        self.scaler = None

    def fit(self, X_train, y_train):
        """ fit scaler and save X_train and y_train """
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train


    @abstractmethod
    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels """

    def neighbours_indices(self, x):
        """ for a given point x, find indices of k closest points in the training set """
        closest_indices = np.empty((self.x_train.shape[0], 2))
        for rows in range(closest_indices.shape[0]):
            closest_indices[rows, 0] = self.dist(x, self.x_train[rows])
            closest_indices[rows, 1] = self.y_train[rows]


        return_array = closest_indices[np.argsort(closest_indices[:, 0])]
        return return_array[:self.k, 1] # returns a new np array sized k of sorted distances

    @staticmethod
    def dist(x1, x2):
        """returns Euclidean distance between x1 and x2"""
        return np.linalg.norm(x1 - x2) # Euclidian distance between the two vectors


class RegressionKNN(KNN):
    def __init__(self, k):
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels by regression """
        y_pred = []
        X_test = self.scaler.transform(X_test)
        for rows in range(X_test.shape[0]):
            closest = self.neighbours_indices(X_test[rows])
            y_pred.append(closest.mean())

        return np.array(y_pred)


class ClassificationKNN(KNN):
    def __init__(self, k):
        super().__init__(k)

    def predict(self, X_test):
        """ predict labels for X_test and return predicted labels by regression """
        y_pred = []
        X_test = self.scaler.transform(X_test)
        for rows in range(X_test.shape[0]):
            closest = self.neighbours_indices(X_test[rows])
            y_pred.append(mode(closest)) # uses the mode function from statistics

        return np.array(y_pred)
