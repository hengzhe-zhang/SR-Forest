from abc import abstractmethod

import math
import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


class EnsembleSR(BaseEstimator, RegressorMixin):
    def __init__(self, sr_model, decision_tree=None, **kwargs):
        self.kwargs = kwargs
        self.sr_model = sr_model
        self.decision_tree = decision_tree
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
        est = self.sr_model
        est.fit(X, y)
        self.ensemble_size = self.threshold_determination(est, X, y)
        predictions = self.top_predictions(est, X, self.ensemble_size)
        if self.decision_tree is not None:
            for prediction in predictions:
                dt = clone(self.decision_tree)
                dt.fit(X, y - prediction)
                self.dt_list.append(dt)
        return self

    def predict(self, X):
        if self.decision_tree is not None:
            predictions = self.top_predictions(self.sr_model, X, self.ensemble_size)
            for id, prediction in enumerate(predictions):
                predictions[id] = prediction + self.dt_list[id].predict(X)
        else:
            predictions = self.top_predictions(self.sr_model, X, self.ensemble_size)
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