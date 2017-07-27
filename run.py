import gc
from zipfile import ZipFile
from datetime import datetime

import pandas as pd
import numpy as np

from debug import logger
from load import loadNclean
from models import build_stack, classify_outliners

df_train, prop = loadNclean()

logger.debug('Training outliners classifier')
rf_clf, df_train = classify_outliners(df_train)
train_in = df_train[df_train.outliner == 0].drop('outliner', axis=1)
train_out = df_train[df_train.outliner == 1].drop('outliner', axis=1)

logger.debug('Building regression model for inliners')
y = train_in.logerror
x = train_in.drop(['parcelid','logerror'], axis=1)
in_stack = build_stack(x, y)

logger.debug('Building regression model for outliners')
y = train_out.logerror
x = train_out.drop(['parcelid','logerror'], axis=1)
out_stack = build_stack(x, y)

logger.debug('Clean up training data')
del x, y, df_train, train_in, train_out
gc.collect()

logger.debug('Loading submission template')
with ZipFile('../sample_submission.csv.zip') as zipped:
    sub = pd.read_csv(zipped.open('sample_submission.csv'))

df_sub = pd.DataFrame(sub['ParcelId'].values, columns=['parcelid'])
df_sub = pd.merge(df_sub, prop, on='parcelid')
for i in range(1, 13, 1):
    df_sub["month_%i"%i] = 0

logger.debug('classifying outliners')
outliners = rf_clf.predict(df_sub.drop('parcelid', axis=1))
df_sub['outliner'] = outliners
x_sub_in = df_sub[df_sub.outliner == 0].drop('outliner', axis=1)
x_sub_out = df_sub[df_sub.outliner == 1].drop('outliner', axis=1)
# del x_sub, rf_clf
# gc.collect()

#predict for each month
logger.debug('Running predictions')
for c in sub.columns[sub.columns != "ParcelId"]:
    date = datetime.strptime(c, "%Y%m")
    logger.debug('Calculating for %s' % date.strftime("%b %y"))
    logger.debug('Inliners')
    for i in range(1, 13, 1):
        x_sub_in["month_%i"%i] = 0
    x_sub_in['month_%i'%date.month] = 1
    x_sub_in['logerror'] = in_stack.predict(x_sub_in.drop('parcelid', axis=1))
    logger.debug('Outliners')
    for i in range(1, 13, 1):
        x_sub_out["month_%i"%i] = 0
    x_sub_out['month_%i'%date.month] = 1
    x_sub_out['logerror'] = in_stack.predict(x_sub_out.drop('parcelid', axis=1))
    sub_cat = pd.concat([x_sub_in, x_sub_out])
    df_sub[c] = pd.merge(df_sub, sub_cat, on='parcelid').logerror
    x_sub_in.drop('logerror', axis=1, inplace=True)
    x_sub_out.drop('logerror', axis=1, inplace=True)
    sub[c] = df_sub[c]
#
logger.debug('Saving submition data')
sub.to_csv('submit.gz', compression='gzip', index=False, float_format='%.4f')
