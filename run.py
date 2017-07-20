import gc
from zipfile import ZipFile

import pandas as pd

from debug import logger
from load import loadNclean, simple_load_train
from models import build_rf

df_train, prop = loadNclean()
y = df_train.logerror
x = df_train.drop(['parcelid','logerror'], axis=1)
feature_list = x.columns
est, Y_test, Y_pred = build_rf(x, y)

logger.debug('Clearing training data from memory')
del df_train; del y; del x; gc.collect()

logger.debug('loading submission template')
with ZipFile('sample_submission.csv.zip') as zipped:
    sub = pd.read_csv(zipped.open('sample_submission.csv'))

df_sub = pd.DataFrame(sub['ParcelId'].values, columns=['parcelid'])
df_sub = pd.merge(df_sub, prop, on='parcelid')
x_sub = df_sub.drop('parcelid', axis=1)

logger.debug('Running estimator on required parcels')
sub_pred = est.predict(x_sub)

if len(sub_pred) != len(sub):
    raise(ValueError('Lengths not equal, missing props'))

logger.debug('Saving submition data')
for c in sub.columns[sub.columns != "ParcelId"]:
    sub[c] = sub_pred

sub.to_csv('submit.csv', index=False, float_format='%.4f')
