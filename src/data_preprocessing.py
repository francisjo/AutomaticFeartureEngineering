from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import category_encoders as ce

class PreprocessingData:

    # Transform the categorical data by different encoding methods #
    def categorical_transformer_method(categorical_method):
        simple_imputer = SimpleImputer(strategy="most_frequent", copy=False)
        if categorical_method == 'BinaryEncoder':
            categorical_transformer = Pipeline(
                steps=
                [
                    ('imputer', simple_imputer),
                    ('BinaryEncoder', ce.BinaryEncoder())
                ]
            )
        elif categorical_method == 'OneHotEncoder':
            categorical_transformer = Pipeline(
                steps=
                [
                    ('imputer', simple_imputer),
                    ('onehot', ce.OneHotEncoder())
                ]
            )
        elif categorical_method == 'HashingEncoder':
            categorical_transformer = Pipeline(
                steps=
                [
                    ('imputer', simple_imputer),
                    ('onehot', ce.HashingEncoder())
                ]
            )
        return categorical_transformer
