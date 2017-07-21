from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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
    logger.debug('Starting Grid search')
    est = RandomForestRegressor(random_state=SEED)
    params_grid = {
            "n_estimators" : [100, 200, 300],
            "max_features" : [10, 25],
            "min_samples_leaf" : [1, 5, 10]}
    grid = GridSearchCV(est, params_grid, verbose=2)
    grid.fit(x_train, y_train)
    print(grid.best_params_)

    logger.debug('Building estimator')
    est.set_params(**grid.best_params_)
    est.fit(x_train, y_train)
    logger.debug('Making predictions')
    y_pred = est.predict(x_test)
    logger.debug('Scoring test')
    print("Model score: {:0.3f}".format(est.score(x_test, y_test)))
    print("Mean abs error: {:0.3f}".format(mean_absolute_error(y_test, y_pred)))
    return est, y_test, y_pred
