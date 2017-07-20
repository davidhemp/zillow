from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from debug import logger

def build_rf(x, y):
    SEED = 148
    logger.debug('Splitting data')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

    est = RandomForestRegressor(random_state=SEED,
                                n_estimators=100,
                                max_features=50,
                                min_samples_leaf=10)
    logger.debug('Building estimator')
    est.fit(x_train, y_train)
    logger.debug('Making predictions')
    y_pred = est.predict(x_test)
    logger.debug('Scoring test')
    print("Model score: {:0.3f}".format(est.score(x_test, y_test)))
    print("Mean abs error: {:0.3f}".format(mean_absolute_error(y_test, y_pred)))
    return est, y_test, y_pred
