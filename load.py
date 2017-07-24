# coding: utf-8
from zipfile import ZipFile
from debug import logger

import pandas as pd
import numpy as np

## Loader from zips, kept as zips just for to play with ZipFile
def loadNclean():
    logger.debug('Loading properties data')
    with ZipFile('../properties_2016.csv.zip') as zipped:
        prop = pd.read_csv(zipped.open('properties_2016.csv'), low_memory=False)

    logger.debug('Loading train data')
    with ZipFile('../train_2016_v2.csv.zip') as zipped:
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
    prop.drop(  ['architecturalstyletypeid',
                'assessmentyear',
                'propertycountylandusecode'],
                axis=1,
                inplace=True)

    ## changed to binary style, done this so I can see exactly what is changed how
    logger.debug(r'Switching from true/null to binary')
    change_list = [ 'hashottuborspa',
                    'threequarterbathnbr',
                    'calculatedbathnbr',
                    'fireplaceflag',
                    'basementsqft',
                    'taxdelinquencyflag',
                    'airconditioningtypeid',
                    'decktypeid']
    for c in change_list:
        prop[c] = prop[c].notnull().astype('int')

    #bedroomcnt 0 to 10 and 11+
    # prop.loc[prop.bedroomcnt > 10, 'bedroomcnt'] = 11

    # location information
    logger.debug('Building location categories')
    prop['fips'].replace(np.NaN, 0, inplace=True)

    j = 65
    for i in prop.fips.unique():
        prop.loc[prop.fips == i, 'fips'] = chr(j)
        j += 1
    df_area = pd.get_dummies(prop.fips, prefix='area')
    prop = pd.concat([prop, df_area], axis=1)
    prop.drop('fips', axis=1, inplace=True)
    prop.drop('rawcensustractandblock', axis=1, inplace=True)
    prop.drop('propertyzoningdesc', axis=1, inplace=True)

    #month information
    logger.debug('Building month categories')
    train['date'] = pd.to_datetime(train.transactiondate)
    train['month'] = train.date.map(lambda x: x.month)
    # df_month = pd.get_dummies(prop.month, prefix='month')
    # print(df_month.columns)
    # prop = pd.concat([prop, df_month], axis=1)
    train.drop(['transactiondate', 'date'], axis=1, inplace=True)

    ## join train and prop
    nulllist = prop.columns[prop.isnull().any()].tolist()
    print(nulllist)
    prop.fillna(0, inplace=True)
    df_train = pd.merge(prop, train, on='parcelid')
    return  df_train, prop
