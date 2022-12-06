import random

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sr_forest.operon_forest import OperonX, OperonForest

np.random.seed(0)
random.seed(0)

X, y = fetch_openml(data_id=1089, return_X_y=True)
X = StandardScaler().fit_transform(X)
X, y = np.array(X), np.array(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
simple_operon = []
for _ in range(20):
    e = OperonX(generations=100, population_size=100)
    e.fit(x_train, y_train)
    print(r2_score(y_train, e.predict(x_train)))
    print(r2_score(y_test, e.predict(x_test)))
    simple_operon.append(r2_score(y_test, e.predict(x_test)))

# operon_forest = []
# for _ in range(20):
#     e = OperonForest(OperonX(generations=100, population_size=100))
#     e.fit(x_train, y_train)
#     print('Ensemble Size', len(e.selected_index))
#     print(r2_score(y_train, e.predict(x_train)))
#     print(r2_score(y_test, e.predict(x_test)))
#     operon_forest.append(r2_score(y_test, e.predict(x_test)))
#
# operon_sr_forest = []
# for _ in range(20):
#     e = OperonForest(OperonX(generations=100, population_size=100),
#                      decision_tree=DecisionTreeRegressor(splitter='random'))
#     e.fit(x_train, y_train)
#     print('Ensemble Size', len(e.selected_index))
#     print(r2_score(y_train, e.predict(x_train)))
#     print(r2_score(y_test, e.predict(x_test)))
#     operon_sr_forest.append(r2_score(y_test, e.predict(x_test)))

print('A', np.mean(simple_operon))
# print('B', np.mean(operon_forest))
# print('C', np.mean(operon_sr_forest))
