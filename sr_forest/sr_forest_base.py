from abc import abstractmethod

import math
import numpy as np
from sklearn import clone
from sklearn.metrics import r2_score


class EnsembleSR():
    def __init__(self, sr_model=None, decision_tree=None, **kwargs):
        self.kwargs = kwargs
        self.est = sr_model
        self.dt = decision_tree
        self.dt_list = []
        self.ensemble_size = 0
        self.selected_index = None

    @abstractmethod
    def top_predictions(self, est, X, ensemble_size):
        pass

    @abstractmethod
    def threshold_determination(self, est, X, y):
        pass

    def fit(self, X, y):
        est = self.est
        est.fit(X, y)
        self.ensemble_size = self.threshold_determination(est, X, y)
        predictions = self.top_predictions(est, X, self.ensemble_size)
        if self.dt is not None:
            for prediction in predictions:
                dt = clone(self.dt)
                dt.fit(X, y - prediction)
                self.dt_list.append(dt)
        return self

    def predict(self, X):
        if self.dt is not None:
            predictions = self.top_predictions(self.est, X, self.ensemble_size)
            for id, prediction in enumerate(predictions):
                predictions[id] = prediction + self.dt_list[id].predict(X)
        else:
            predictions = self.top_predictions(self.est, X, self.ensemble_size)
        prediction = np.mean(predictions, axis=0).flatten()
        return prediction

    def ensemble_selection(self, fitness_values, predictions, y):
        best_size = 0
        best_score = -math.inf
        for size in range(10, 110, 10):
            selected_index = np.argsort(fitness_values)[:size]
            prediction = np.mean([predictions[i] for i in selected_index], axis=0)
            if r2_score(y, prediction) > best_score:
                best_score = r2_score(y, prediction)
                self.selected_index = selected_index
                best_size = size
        return best_size
