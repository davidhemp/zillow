from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.metrics import mean_absolute_error, classification_report

import numpy as np
import pandas as pd

from debug import logger

def classify_outliners(train):
    no_std = 3
    logger.debug('Outliners defined as +/- %0.1f std' %no_std)
    std = no_std*train.logerror.std()
    train['outliner'] = 0
    train.loc[train.logerror > std, 'outliner']=1
    train.loc[train.logerror < -std, 'outliner']=1

    logger.debug('Building balanced train set')
    df_train = train[train.outliner == 0].sample(n=1330)
    df_train = pd.concat([df_train, train[train.outliner==1]])
    df_train.drop(['parcelid', 'logerror'],
                    axis=1,
                    inplace=True)
    x = df_train.drop('outliner', axis=1)
    y = df_train.outliner
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    logger.debug('Training RF classifier')
    clf = RandomForestClassifier(random_state=40)
    params = {  'min_samples_leaf'  : 10,
                'n_estimators'      : 50,
                'max_features'      : None,
                'min_samples_split' : 2,
                'max_depth'         : 10}
    clf.set_params(**params)
    clf.fit(x_train, y_train)

    logger.debug('Testing classifier')
    y_true, y_pred = y_test, clf.predict(x_test)
    y_pred_probs = clf.predict_proba(x_test)
    print(classification_report(y_true, y_pred))
    return clf, train

class Stacker():
    def fit(self, x, y):
        SEED = 148
        pred_data = []
        param_dict = {'n_estimators': 300,
                        'max_features': 10,
                        'min_samples_leaf': 10}
        logger.debug('Building RandomForest model')
        self.rf_est = RandomForestRegressor(**param_dict, random_state=SEED)
        self.rf_est.fit(x, y)
        logger.debug('Building LinearRegression model')
        self.lr_est = LinearRegression()
        self.lr_est.fit(x, y)

    def predict(self, x):
        pred_data = [self.rf_est.predict(x)]
        logger.info("Std for RandomForest: %0.4f" %np.std(pred_data[0]))
        pred_data.append(self.lr_est.predict(x))
        logger.info("Std for LinearRegression: %0.4f" %np.std(pred_data[0]))
        df_pred = pd.DataFrame(pred_data)
        return df_pred.mean(axis=0)

def build_stack(x,y):
    RES = 10
    SEED = 148
    logger.debug('Splitting data')
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=SEED)

    logger.debug('Building stack')
    stack_model = Stacker()
    stack_model.fit(x_train, y_train)
    logger.debug('Making predictions')
    y_pred = stack_model.predict(x_test)
    logger.info("Mean abs error: {:0.5f}".format(mean_absolute_error(y_test, y_pred)))
    return stack_model
