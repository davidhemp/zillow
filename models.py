from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
import numpy as np

from debug import logger

def build_rf(x, y):
    SEED = 148
    RES = 10
    logger.debug('Splitting data')
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=SEED)
    logger.debug('Building train/test estimator')
    param_dict = {'n_estimators': 300,
                    'max_features': 10,
                    'min_samples_leaf': 10}
    est = RandomForestRegressor(random_state=SEED, **param_dict)
    est.fit(x_train, y_train)
    logger.debug('Making predictions')
    y_pred = est.predict(x_test)
    logger.debug('Scoring test')
    print("Model score: {:0.3f}".format(est.score(x_test, y_test)))
    print("Mean abs error: {:0.3f}".format(mean_absolute_error(y_test, y_pred)))
    # return using full data
    logger.debug('Building full estimator')
    est_full = RandomForestRegressor(random_state=SEED, **param_dict)
    est_full.fit(x, y)
    return est_full

def build_linear(x, y):
    SEED = 148
    RES = 10
    logger.debug('Splitting data')
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=SEED)
    logger.debug('Building train/test estimator')
    est = LinearRegression()
    params_grid = {'fit_intercept':[True,False],
                    'normalize':[True,False],
                    'copy_X':[True, False]}
    logger.debug('Grid search for params')
    grid = GridSearchCV(est, params_grid, verbose=1)
    grid.fit(x_train, y_train)
    print(grid.best_params_)
    logger.debug('Building estimator')
    est.set_params(**grid.best_params_)
    est.fit(x_train, y_train)
    logger.debug('Making predictions')
    y_pred = est.predict(x_test)
    logger.debug('Scoring test')
    print("Model score: {:0.5f}".format(est.score(x_test, y_test)))
    print("Mean abs error: {:0.5f}".format(mean_absolute_error(y_test, y_pred)))
    est_full = LinearRegression(**grid.best_params_)
    est_full.fit(x, y)
    return est_full
