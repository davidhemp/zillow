import gc
from zipfile import ZipFile
from datetime import datetime

import pandas as pd
import numpy as np

from debug import logger
from load import loadNclean
from models import build_stack

df_train, prop = loadNclean()

no_std = 3
logger.debug('Outliners defined as +/- %0.1f std' %no_std)
cut_off = no_std*df_train.logerror.std()
df_train = df_train[(df_train.logerror < cut_off) & (df_train.logerror > -cut_off)]

logger.debug('Building stack for inliners only')
y = df_train.logerror
x = df_train.drop(['parcelid','logerror'], axis=1)
model = build_stack(x, y)

# logger.debug('Clean up training data')
# del x, y, df_train
# gc.collect()

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
    sub_pred = model.predict(x_sub)
    sub[c] = sub_pred

logger.debug('Saving submition data')

sub.to_csv('submit.gz', compression='gzip', index=False, float_format='%.4f')
