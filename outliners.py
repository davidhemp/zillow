# coding: utf-8
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_curve

# load train data
train = pd.read_csv('train.csv')

# clean data
train.drop(train.columns[0], axis=1, inplace=True)
for c in train.dtypes[train.dtypes==object].index.values:
    train[c] = (train[c] == True)

train.fillna(0, inplace=True)

# define outliners as 3 std
std = 3*train.logerror.std()
train['outliner'] = 0
train.loc[train.logerror > std, 'outliner']=1
train.loc[train.logerror < -std, 'outliner']=1

#build balanced train dataframe for classifier
df_train = train[train.outliner == 0].sample(n=1330)
df_train = pd.concat([df_train, train[train.outliner==1]])
df_train.drop(['parcelid', 'transactiondate', 'logerror', 'assessmentyear'],
                axis=1,
                inplace=True)
x = df_train.drop('outliner', axis=1)
y = df_train.outliner
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = RandomForestClassifier(random_state=40)
# clf = GradientBoostingClassifier()
# param_grid = { "n_estimators"       : [25, 50],
#             #    "criterion"          : ["gini", "entropy"],
#                "max_depth"          : [1, 5, 10],
#                "min_samples_split"  : [2, 4] ,
#                "max_features"       : ["sqrt", "log2", None],
#             #    "oob_score"          : [True, False],
#                "min_samples_leaf"   : [1, 5, 10]}
#
# grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=3, verbose=1)
# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
# clf.set_params(**grid_search.best_params_)
params = {  'min_samples_leaf'  : 10,
            'n_estimators'      : 50,
            'max_features'      : None,
            'min_samples_split' : 2,
            'max_depth'         : 10}
clf.set_params(**params)
clf.fit(x_train, y_train)
y_true, y_pred = y_test, clf.predict(x_test)
y_pred_probs = clf.predict_proba(x_test)
print(classification_report(y_true, y_pred))
