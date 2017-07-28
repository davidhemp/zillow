from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Lars
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

from debug import logger
from model_params import lvl1_params

class Stacker():
    def fit(self, x, y):
        SEED = 148
        logger.debug('Training base models')
        kf = KFold(n_splits=5)
        debug_strs = [  'Building RandomForest model',
                        'Building AdaBoostRegressor model',
                        'Building GradientBoosting model',
                        'Building XGBRegressor model',
                        'Building BayesianRidge model']
        self.models = [RandomForestRegressor(**lvl1_params[0], random_state=SEED),
                        AdaBoostRegressor(),
                        GradientBoostingRegressor(),
                        XGBRegressor(),
                        BayesianRidge()]
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=SEED)
        x_meta = np.zeros((len(self.models), len(y_test)))
        for i, model in enumerate(self.models):
            logger.debug(debug_strs[i])
            model.fit(x_train, y_train)
            x_meta[i] += model.predict(x_test)
        logger.debug('Training meta model')
        self.meta = LinearRegression()
        self.meta.fit(x_meta.T, y_test)

    def predict(self, x):
        debug_strs = [  'Predicting with RandomForest model',
                        'Predicting with AdaBoostRegressor model',
                        'Predicting with GradientBoosting model',
                        'Predicting with XGBRegressor model',
                        'Predicting with BayesianRidge model']
        x_meta = np.zeros((len(self.models), len(x)))
        for i, model in enumerate(self.models):
            logger.debug(debug_strs[i])
            x_meta[i] += model.predict(x)
        logger.debug('Predicting with meta model')
        return self.meta.predict(x_meta.T)

def build_stack(x, y):
    SEED = 48
    logger.debug('Building stack')
    stack_model = Stacker()
    x_train, x_hold_out, y_train, y_hold_out = \
            train_test_split(x, y, test_size=0.2, random_state=SEED)
    stack_model.fit(x_train, y_train)
    logger.debug('Testing stack')
    y_pred = stack_model.predict(x_hold_out)
    abs_error = mean_absolute_error(y_hold_out, y_pred)
    logger.info("Mean abs error: {:0.5f}".format(abs_error))
    return stack_model
