from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import scipy
import scipy.stats


class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, X, y):
        self.x_train = np.copy(X)
        self.y_train = np.copy(y)
        return self

    def predict(self, X):
        predictions = []
        distX = scipy.spatial.distance.cdist(X, self.x_train, 'euclidean')
        distSelection = np.argpartition(a=distX, axis=-1, kth=self.n_neighbors)
        firstKidx = distSelection[:, :self.n_neighbors]
        firstKlabels = np.apply_along_axis(func1d=(lambda idx: scipy.stats.mode((self.y_train[idx]))[0][0]), axis=-1,
                                           arr=firstKidx)
        predictions = firstKlabels
        return predictions
