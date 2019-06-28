import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns  # list of column to encode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = pd.DataFrame(X.copy())
        for colname, col in output.iteritems():
            encoding = output.groupby(colname).size()
            encoding = encoding / len(output)
            output[colname] = output[colname].map(encoding)

        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)