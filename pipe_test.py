""" I was going to use this method but a number of base models require
special transforms for fit so just going back to custom stack model"""

from sklearn.base import TransformerMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

# from models import make_model

class RidgeTransformer(Ridge, TransformerMixin):
    def transform(self, X, *_):
        return self.predict(X)

class RandomForestTransformer(RandomForestRegressor, TransformerMixin):
    def transform(self, X, *_):
        print("Transforming rf")
        return self.predict(X)

class KNeighborsTransformer(KNeighborsRegressor, TransformerMixin):
    def transform(self, X, *_):
        return self.predict(X)

class GradientBoostTransformer(GradientBoostingRegressor, TransformerMixin):
    def transform(self, x, *_):
        print("Transforming grad boost")
        return self.predict(x)

def build_model():
    pred_union = FeatureUnion(
        transformer_list=[
            ('grad_boost', GradientBoostTransformer()),
            ('rand_forest', RandomForestTransformer())
        ],
        n_jobs=1
    )
    model = Pipeline(steps=[
        ('pred_union', pred_union),
        ('lin_regr', LinearRegression())
    ])

    return model

print('Build and fit a model..')

model = build_model()

# model = make_model()

X, y = make_regression(n_samples=10000, n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("beep")
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print('Done. Score:', score)
