from pyGPGOMEA import GPGOMEARegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from sr_forest.gp_gomea_forest import GPGOMEAForest

X, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
gp = GPGOMEAForest(GPGOMEARegressor(generations=10), decision_tree=DecisionTreeRegressor(splitter='random'))
r = GridSearchCV(gp, {
    'sr_model': [
        GPGOMEARegressor(generations=5),
        GPGOMEARegressor(generations=20),
    ]
}, n_jobs=-1, cv=3)
r.fit(x_train, y_train)
print(r2_score(y_test, r.predict(x_test)))
