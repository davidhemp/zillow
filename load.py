# coding: utf-8
from zipfile import ZipFile
from debug import logger

import pandas as pd
import numpy as np

## Loader from zips, kept as zips just for to play with ZipFile
def loadNclean():
    logger.debug('loading properties data')
    with ZipFile('properties_2016.csv.zip') as zipped:
        prop = pd.read_csv(zipped.open('properties_2016.csv'), low_memory=False)

    logger.debug('loading train data')
    with ZipFile('train_2016_v2.csv.zip') as zipped:
        train = pd.read_csv(zipped.open('train_2016_v2.csv'))

    # prop.head()
    # train.head()
    #reduce load size
    logger.debug('Changing down to 32 bit floats')
    for c, dtype in zip(prop.columns, prop.dtypes):
        if dtype == np.float64:
            prop[c] = prop[c].astype(np.float32)

    ## clean data, remove NaN
    # Remove data that isn't useful, only a few of these anyway
    prop.drop('architecturalstyletypeid', axis=1, inplace=True)
    prop.drop('assessmentyear', axis=1, inplace=True)
    prop.drop('propertycountylandusecode', axis=1, inplace=True)

    ## changed to binary style
    logger.debug('Cleaning hottub/spa data')
    prop['hottub'] = prop['hashottuborspa'].notnull().astype('int')
    prop.drop('hashottuborspa', axis=1, inplace=True)

    logger.debug('cleaning bath room data')
    prop['threequarterbathnbr'].replace(np.NaN, 0, inplace=True)
    prop['calculatedbathnbr'].replace(np.NaN, 0, inplace=True)
    prop.drop('bathroomcnt', axis=1, inplace=True)

    logger.debug('cleaning fire place data')
    prop['fireplace'] = prop['fireplaceflag'].notnull().astype('int')
    prop.drop('fireplaceflag', axis=1, inplace=True)

    logger.debug('cleaning basement data')
    prop['basement'] = prop['basementsqft'].notnull().astype('int')
    prop.drop('basementsqft', axis=1, inplace=True)

    logger.debug('cleaning tax delinquency data')
    prop['taxdelinquencyflag'] = prop['taxdelinquencyflag'].notnull().astype('int')

    logger.debug('changing aircon data to binary')
    prop['aircon'] = prop['airconditioningtypeid'].notnull().astype('int')
    prop.drop('airconditioningtypeid', axis=1, inplace=True)

    logger.debug('changing deck type to binary')
    #have no data for other deck ids so binary
    prop['deck'] = prop['decktypeid'].notnull().astype('int')
    prop.drop('decktypeid', axis=1, inplace=True)

    #bedroomcnt 0 to 10 and 11+
    # prop.loc[prop.bedroomcnt > 10, 'bedroomcnt'] = 11

    # location information
    prop.drop('rawcensustractandblock', axis=1, inplace=True)
    prop.drop('propertyzoningdesc', axis=1, inplace=True)
    prop['fips'].replace(np.NaN, 0, inplace=True)

    j = 65
    for i in prop.fips.unique():
        prop.loc[prop.fips == i, 'fips'] = chr(j)
        j += 1
    df_area = pd.get_dummies(prop.fips, prefix='area')
    prop = pd.concat([prop, df_area], axis=1)
    prop.drop('fips', axis=1, inplace=True)

    #month information?
    train.drop('transactiondate', axis=1, inplace=True)

    ## join train and prop
    prop.fillna(0, inplace=True)
    df_train = pd.merge(prop, train, on='parcelid')
    return  df_train, prop

def simple_load_train():
    """Read in training data and return input, output, columns tuple."""

    # This is a version of Anovas minimally prepared dataset
    # for the xgbstarter script
    # https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655

    logger.debug('loading properties')
    with ZipFile('properties_2016.csv.zip') as zipped:
        prop = pd.read_csv(zipped.open('properties_2016.csv'), low_memory=False)

    logger.debug('loading train results')
    with ZipFile('train_2016_v2.csv.zip') as zipped:
        train = pd.read_csv(zipped.open('train_2016_v2.csv'))

    convert = prop.dtypes == 'float64'
    prop.loc[:, convert] = \
        prop.loc[:, convert].apply(lambda x: x.astype(np.float32))

    df = train.merge(prop, how='left', on='parcelid')

    y = df.logerror
    df = df.drop(['transactiondate',
                  'propertyzoningdesc',
                  'taxdelinquencyflag',
                  'propertycountylandusecode'], axis=1)

    convert = df.dtypes == 'object'
    df.loc[:, convert] = \
        df.loc[:, convert].apply(lambda x: 1 * (x == True))

    df.fillna(0, inplace=True)

    return df, prop
