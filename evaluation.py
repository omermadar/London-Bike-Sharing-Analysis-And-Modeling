import numpy as np
import matplotlib.pyplot as plt

def f1_score(y_true, y_pred):
    """
    returns f1_score of binary classification task with true labels y_true and predicted labels y_pred
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = ((2 * precision * recall) / (precision + recall))

    return F1


def rmse (y_true, y_pred):
    """
    returns RMSE of regression task with true labels y_true and predicted labels y_pred
    """
    RMSE = np.sqrt(np.mean((y_true - y_pred)**2))
    return RMSE

def visualize_results(k_list, scores, metric, title, path):
    """
     plot a results graph of cross validation scores
    """
    x = k_list
    y = scores
    plt.plot(x, y)
    plt.xlabel('$k$')
    plt.ylabel(metric)
    plt.title(title)
    plt.savefig(path)
    plt.close()