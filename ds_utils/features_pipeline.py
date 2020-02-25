import logging
import re
from collections import Counter, OrderedDict
import numpy as np
from pandas import DataFrame
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler, PolynomialFeatures,  MinMaxScaler

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if DataFrame is type(X):
            return X[self.key]
        else:
            raise Exception("unsupported itemselector type. implement some new stuff: %s" % type(X))

class Reshaper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:,None]

class Dummyizer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.dummyizer = LabelBinarizer()
        self.dummyizer.fit(X)
        return self

    def transform(self, X):
        return self.dummyizer.transform(X)

class Concatenator(BaseEstimator, TransformerMixin):
    def __init__(self, glue=" "):
        self.glue = glue

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = len(list(X.shape))
        out = ["%s" % (self.glue.join(x) if cols > 1 else x) for x in X]
        return out
            
class Floater(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype("float64")

class Densinator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.todense()

class Quantiler(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=100):
        self.n_quantiles = n_quantiles
    def fit(self, X, y=None):
        percentiles = np.linspace(0, 100, self.n_quantiles+2)
        self.quantiles = np.percentile(X, percentiles)
        return self

    def find_quantile(self, x):
        return [1 if self.quantiles[i] < x and self.quantiles[i+1] >= x else 0 for i in range(0, len(self.quantiles) - 1)]
        
    def transform(self, X):
        return [self.find_quantile(x) for x in X]

def build_poly_wrapper(col,
                       degree=2,
                       transformers=[]):

    transformer_list = [get_transformer(trans['name'])(col, **trans['config']) for trans in transformers]
    return (col + '_poly', Pipeline([
           ('union', FeatureUnion(transformer_list=transformer_list)),
           ('densinator', Densinator()),
           ('poly', PolynomialFeatures(degree=degree))
    ]))

def build_numeric_column(col):
    return ("numeric_%s" % col, Pipeline([
                ('selector', ItemSelector(col)), 
                ('reshaper', Reshaper()),
                ('floater', Floater()),
                ('scaler', StandardScaler())]))

def build_quantile_column(col,
                          n_quantiles=100):
    return ("quantile_%s" % col, Pipeline([
                ('selector', ItemSelector(col)), 
                ('reshaper', Reshaper()),
                ('quantiler', Quantiler(n_quantiles))]))

def build_range_scaler(col,
                       min=0,
                       max=1):
    return ("min_max %s" % col, Pipeline([
                ('selector', ItemSelector(col)),
                ('reshaper', Reshaper()),
                ('min_max', MinMaxScaler(feature_range=(min, max)))]))

def build_dummyizer(col):
     return ("onehot_s_%s" % col, Pipeline([
           ('selector', ItemSelector(col)),
           ('concat_cols', Concatenator()),
           ('label', Dummyizer())]))

def build_null(col):
    return ("null_%s" % col, Pipeline([
                ('selector', ItemSelector(col)), 
                ('reshaper', Reshaper())]))

def get_transformer(name):
    transformer_map = {
        "standard_numeric" : build_numeric_column,
        "quantile_numeric" : build_quantile_column,
        "range_numeric"    : build_range_scaler, 
        "poly"             : build_poly_wrapper,
        "dummyizer"        : build_dummyizer,
        "null_transformer" : build_null
    }
    return transformer_map[name]

def transformer_from_config(field, transformer_config):
    name = transformer_config['name']
    configs = transformer_config.get('config', {})
    return get_transformer(name)(field, **configs)    
    
def pipeline_from_config_file(filename):
    return pipeline_from_config(json.load(open(filename, 'r')))

def pipeline_from_config(configuration):
    transformers = [[transformer_from_config(field_config['field'], transformer_config) for transformer_config in field_config['transformers']] for field_config in configuration]
    transformer_list = reduce(lambda x,y: x+y, transformers)
    return Pipeline([('union', FeatureUnion(transformer_list=transformer_list))])
