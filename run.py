import gc
from zipfile import ZipFile
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from debug import logger
from load import loadNclean
from models import build_rf, build_linear

df_train, prop = loadNclean()

#remove outliners, 3 std is used
std = df_train.logerror.std()
med = df_train.logerror.median()
cutoff = 3*df_train.logerror.std()
df_train = df_train[df_train.logerror < 3*(std+med)]

y = df_train.logerror
x = df_train.drop(['parcelid','logerror'], axis=1)
feature_list = x.columns

est = build_linear(x, y)

logger.debug('Clearing training data from memory')
# del df_train; del y; del x; gc.collect()

logger.debug('loading submission template')
with ZipFile('../sample_submission.csv.zip') as zipped:
    sub = pd.read_csv(zipped.open('sample_submission.csv'))

df_sub = pd.DataFrame(sub['ParcelId'].values, columns=['parcelid'])
df_sub = pd.merge(df_sub, prop, on='parcelid')
x_sub = df_sub.drop('parcelid', axis=1)
for i in range(1, 13, 1):
    x_sub["month_%i"%i] = 0

#predict for each month
logger.debug('Running predictions')
for c in sub.columns[sub.columns != "ParcelId"]:
    for i in range(1, 13, 1):
        x_sub["month_%i"%i] = 0
    date = datetime.strptime(c, "%Y%m")
    x_sub['month_%i'%date.month] = 1
    logger.debug('Calculating for %s' % date.strftime("%b %y"))
    sub_pred = est.predict(x_sub)
    sub[c] = sub_pred

logger.debug('Saving submition data')

sub.to_csv('submit.gz', compression='gzip', index=False, float_format='%.4f')
