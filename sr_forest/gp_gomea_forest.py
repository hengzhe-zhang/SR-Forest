import random

import numpy as np
from pyGPGOMEA import GPGOMEARegressor as GPGR
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sr_forest.sr_forest_base import EnsembleSR


class GPGOMEAForest(EnsembleSR, BaseEstimator, RegressorMixin):
    def __init__(self, decision_tree=None, **kwargs):
        sr_model = GPGR(**kwargs)
        super().__init__(sr_model, decision_tree, **kwargs)

    def threshold_determination(self, est, X, y):
        final_pop = est.get_final_population(X)
        prediction = [x[0] for x in final_pop]
        fitness_values = [p[1] for p in final_pop]
        return self.ensemble_selection(fitness_values, prediction, y)

    def top_predictions(self, est, X, size):
        prediction = [x[0] for x in est.get_final_population(X)]
        prediction = [prediction[i] for i in self.selected_index]
        return prediction


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    X, y = load_diabetes(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X, y = np.array(X), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    e = GPGR(generations=20)
    e.fit(x_train, y_train)
    print(r2_score(y_test, e.predict(x_test)))

    e = GPGOMEAForest(generations=20)
    e.fit(x_train, y_train)
    print(r2_score(y_test, e.predict(x_test)))

    e = GPGOMEAForest(generations=5, decision_tree=DecisionTreeRegressor(splitter='random'))
    e.fit(x_train, y_train)
    print(r2_score(y_test, e.predict(x_test)))
