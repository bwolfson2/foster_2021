import logging
import re
from collections import Counter, OrderedDict
import numpy as np
import chardet
from pandas import DataFrame
from html2text import html2text
import os
import functools

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler, PolynomialFeatures,  MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

class WordCleaner(BaseEstimator, TransformerMixin):

    def decode(self, content):
        str_bytes = str.encode(content)
        charset = chardet.detect(str_bytes)['encoding']
        return str_bytes.decode(encoding=charset, errors='ignore')

    feature_regex_pipe = [
        (r"\|", " "),
        (r"\r\n?|\n", " "),
        (r"[^\x00-\x7F]+", " "),
        (r"\s+", " "),
        (r"https?://\S+", "_url_"),
        (r"\w{,20}[a-zA-Z]{1,20}[0-9]{1,20}", "_wn_"),
        (r"\d+/\d+/\d+", "_d2_"),
        (r"\d+/\d+", "_d_"),
        (r"\d+:\d+:\d+", "_ts_"),
        (r":", " ")
    ]
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def _text_clean(x):
            all_clean = html2text(self.decode(x))
            replaced = functools.reduce(lambda acc, re_rep: re.sub(re_rep[0], re_rep[1], acc), self.feature_regex_pipe, all_clean)
            return " ".join([y for y in replaced.split(" ") if len(y) <= 20])

        return map(_text_clean, X)
    
    

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

def build_wordcount_transformer(col,
                                binary=False,
                                min_df=0.0,
                                ngrams=2):
    return ("wordcount_%s" % col, Pipeline([
           ('selector', ItemSelector(col)),
           ('concat_cols', Concatenator()),
           ('cleaner', WordCleaner()),
           ('tfidf', CountVectorizer(binary=binary, min_df=min_df, decode_error='ignore', ngram_range=(1,ngrams)))]))

def build_tfidf_transformer(col,
                            min_df=0.0,
                            ngrams=2):
    return ("tfidf_%s" % col, Pipeline([
           ('selector', ItemSelector(col)),
           ('concat_cols', Concatenator()),
           ('cleaner', WordCleaner()),
           ('tfidf', TfidfVectorizer(min_df=min_df, decode_error='ignore', ngram_range=(1,ngrams)))]))

def get_transformer(name):
    transformer_map = {
        "standard_numeric" : build_numeric_column,
        "quantile_numeric" : build_quantile_column,
        "range_numeric"    : build_range_scaler, 
        "poly"             : build_poly_wrapper,
        "dummyizer"        : build_dummyizer,
        "null_transformer" : build_null,
        "tfidf"            : build_tfidf_transformer,
        "word_count"       : build_wordcount_transformer
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
    transformer_list = functools.reduce(lambda x,y: x+y, transformers)
    return Pipeline([('union', FeatureUnion(transformer_list=transformer_list))])
