from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Lars
from sklearn.metrics import mean_absolute_error
# from xgboost import XGBRegressor
import numpy as np
import pandas as pd

from debug import logger
from model_params import lvl1_params, rf_grid

class Stacker(RegressorMixin, BaseEstimator):
    def __init__(self, **params):
        SEED = 148
        self.models = { "rf": RandomForestRegressor(),
                        "ab": AdaBoostRegressor(),
                        "gb": GradientBoostingRegressor(),
                        "br": BayesianRidge()}
        self.params = dict()
        for model_key, model in self.models.items():
            for param_key, value in model.get_params().items():
                self.params[model_key + "__" + param_key] = value

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        valid_params = self.get_params()
        for key, value in params.items():
            if key in valid_params:
                self.params[key] = value
            else:
                raise ValueError('%s is not a valid parameter' %key)
        return self

    def fit(self, x, y):
        SEED = 148
        self.meta = LinearRegression()
        logger.debug('Training base models')
        kf = KFold(n_splits=5)
        debug_strs = [  'Building RandomForest model',
                        'Building AdaBoostRegressor model',
                        'Building GradientBoosting model',
                        'Building BayesianRidge model']
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=SEED)
        x_meta = np.zeros((len(self.models), len(y_test)))
        i = 0
        for model_key, model in self.models.items():
            logger.debug(debug_strs[i])
            model_params = {}
            for params_key, value in self.params.items():
                key_str = params_key.split("__")
                if key_str[0] == model_key:
                    model_params[key_str[1]] = value
            model.set_params(**model_params)
            model.fit(x_train, y_train)
            x_meta[i] += model.predict(x_test)
            i += 1
        logger.debug('Training meta model')
        self.meta.fit(x_meta.T, y_test)

    def predict(self, x):
        debug_strs = [  'Predicting with RandomForest model',
                        'Predicting with AdaBoost model',
                        'Predicting with GradientBoosting model',
                        'Predicting with BayesianRidge model']
        x_meta = np.zeros((len(self.models), len(x)))
        for i, model in enumerate(self.models.values()):
            logger.debug(debug_strs[i])
            x_meta[i] += model.predict(x)
        logger.debug('Predicting with meta model')
        return self.meta.predict(x_meta.T)


def build_stack(x, y, params=None):
    SEED = 48
    logger.debug('Building stack')
    stack_model = Stacker()
    x_train, x_hold_out, y_train, y_hold_out = \
            train_test_split(x, y, test_size=0.2, random_state=SEED)
    if not params:
        params = find_params(x_train, y_train, stack_model)
        stack_model.set_params(**params)
    stack_model.fit(x_train, y_train)
    logger.debug('Testing stack')
    y_pred = stack_model.predict(x_hold_out)
    abs_error = mean_absolute_error(y_hold_out, y_pred)
    logger.info("Mean abs error: {:0.5f}".format(abs_error))
    return stack_model

def find_params(x, y, model):
    params_grid = {**rf_grid}
    grid = GridSearchCV(model, params_grid, verbose=2, return_train_score=False)
    grid.fit(x, y)
    return grid.best_params_

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    model = Stacker()
    x, y = make_regression(n_samples=10000, n_features=10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    rf_grid = { "rf__n_estimators"       : [25, 50],
                #    "criterion"          : ["gini", "entropy"],
                   "rf__max_depth"          : [1, 5, 10],
                   "rf__min_samples_split"  : [2, 4] ,
                   "rf__max_features"       : ["sqrt", "log2", None],
                #    "oob_score"          : [True, False],
                   "rf__min_samples_leaf"   : [1, 5, 10]}

    params_grid = {**rf_grid}
    grid = GridSearchCV(model, params_grid, verbose=2, return_train_score=False)
    grid.fit(x_train, y_train)
    model.set_params(**grid.best_params_)
    model.fit(x_train, y_train)
    logger.info(model.score(x_test, y_test))
    logger.info(grid.best_params_)
