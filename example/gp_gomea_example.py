from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from pyGPGOMEA import GPGOMEARegressor
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sr_forest.gp_gomea_forest import GPGOMEAForest


def train_and_test(key):
    model, X, y, train_index, test_index = key
    x_train, x_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(x_train, y_train)
    return r2_score(y_test, model.predict(x_test))


def custom_cv(model, x, y):
    cv_scores = Pool().map(train_and_test, [(model, x, y, train_index, test_index)
                                            for train_index, test_index in KFold(n_splits=10).split(X, y)])
    print(cv_scores)
    return cv_scores


X, y = make_friedman1()
X = StandardScaler().fit_transform(X)
X, y = np.array(X), np.array(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model_list = {
    'GP-GOMEA': GPGOMEARegressor(generations=20),
    'Ridge': Ridge(),
    'KNN': Pipeline([
        ('Scaler', StandardScaler()),
        ('KNN', KNeighborsRegressor())
    ]),
    'GPR': Pipeline([
        ('Scaler', StandardScaler()),
        ('GP', GaussianProcessRegressor(kernel=Matern(), normalize_y=True))
    ]),
    'DT': DecisionTreeRegressor(),
    'RF': RandomForestRegressor(n_estimators=100),
    'ET': ExtraTreesRegressor(n_estimators=100),
    'AdaBoost': AdaBoostRegressor(n_estimators=100),
    'GBDT': GradientBoostingRegressor(n_estimators=100),
    'XGBoost': XGBRegressor(n_estimators=100, n_jobs=1),
    'LightGBM': LGBMRegressor(n_estimators=100, n_jobs=1),
    'SR-Forest(GOMEA)': GPGOMEAForest(decision_tree=DecisionTreeRegressor(splitter='random'))
}

all_score = []
detailed_score = []
mean_score = []
for model in ['GP-GOMEA', 'GPR', 'KNN', 'Ridge', 'DT',
              'RF', 'ET', 'AdaBoost', 'GBDT', 'XGBoost', 'LightGBM', 'SR-Forest(GOMEA)']:
    model_instance = model_list[model]
    score = custom_cv(model_instance, X, y)
    all_score.append((model, np.mean(score), *score))
    for s in score:
        detailed_score.append((model, s))
    mean_score.append((model, np.mean(score)))
    print(model, score, np.mean(score))
detailed_score = pd.DataFrame(detailed_score, columns=['Model', 'Score ($R^2$)'])
mean_score = pd.DataFrame(mean_score, columns=['Model', 'Score'])
print(pd.DataFrame(all_score))
sns.set(style='whitegrid')
sns.boxplot(x="Model", y="Score ($R^2$)", data=detailed_score, showfliers=True, palette='vlag', width=0.6)
sns.scatterplot(x="Model", y="Score", data=mean_score, color='black', alpha=0.5, label='Mean Score')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
