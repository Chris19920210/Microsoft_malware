import sys
import math
import numpy as np
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
import pandas as pd

path = os.path.join('media', 'chris', 'Elements', 'finaldata_Microsoft_train')
data = pd.read_csv('/'+path+'/'+'train_dataset.csv')
data2 = pd.read_csv('/media/chris/Elements/finaldata_microsoft_test/test_dataset.csv')


class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)


def main():
    X_train = data.drop(['Id', 'Class'], axis=1)
    y_train = data.loc[:, 'Class']
    X_test = data2.drop('Id', axis=1)
    Id = data2.loc[:, 'Id']
    clf = XGBoostClassifier(
        eval_metric='logloss',
        num_class=9,
        nthread=2,
        eta=0.4,
        num_boost_round=120,
        max_depth=6,
        subsample=1,
        colsample_bytree=1,
        silent=0,
        )

    clf.fit(X_train, y_train)
    prediction = clf.predict_proba(X_test)
    columns = ['Prediction'+str(i) for i in range(1, 10)]
    prediction = pd.DataFrame(prediction, columns=columns)
    results = pd.concat([Id, prediction], axis=1)
    results.to_csv('/media/chris/Elements/finaldata_microsoft_test/test_results.csv', index=False)


if __name__ == '__main__':
    main()
