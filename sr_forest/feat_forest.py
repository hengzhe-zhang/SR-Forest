import numpy as np
from feat import Feat
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sr_forest.sr_forest_base import EnsembleSR


class EnsembleFeat(EnsembleSR):
    def __init__(self, sr_model=None, decision_tree=None):
        super().__init__(sr_model, decision_tree)

    def threshold_determination(self, est, X, y):
        prediction = self.make_prediction(X, est)
        # sorting in ascending order
        fitness_values = [-1 * r2_score(y, p) for p in prediction]
        return self.ensemble_selection(fitness_values, prediction, y)

    def top_predictions(self, est, X, ensemble_size):
        prediction = self.make_prediction(X, est)
        prediction = [prediction[i] for i in self.selected_index]
        return prediction

    def make_prediction(self, X, est):
        prediction = est.predict_archive(X)
        return np.array([x['y_pred'].flatten() for x in prediction])


if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X, y = np.array(X), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    e = RandomForestRegressor(n_jobs=1)
    e.fit(x_train, y_train)
    print(r2_score(y_test, e.predict(x_test)))

    e = EnsembleFeat(Feat(pop_size=200, gens=20))
    e.fit(x_train, y_train)
    print(r2_score(y_test, e.predict(x_test)))

    e = EnsembleFeat(Feat(pop_size=200, gens=20), decision_tree=DecisionTreeRegressor(splitter='random'))
    e.fit(x_train, y_train)
    print(r2_score(y_test, e.predict(x_test)))
