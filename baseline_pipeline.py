import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import column_type_classification as col_classify
import data_cleaning as cd
import category_encoders as ce


# global variables
threshold = 0.20
target_label = 'price'


# load dataset into a pandas data-frame
def load_data():
    titanic = 'https://raw.githubusercontent.com/nphardly/titanic/master/data/inputs/train.csv'
    heart = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/heart.csv'
    car = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car_data.csv'
    wikipedia = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/smallwikipedia.csv'

    df = pd.read_csv(car)
    return df


# passing the data-frame to the run_model() function
def main_func():
    df = load_data()
    result = run_model(df)
    print("Model Score with: %.3f" % result)


# drop the target label from a list
def drop_target_label(cols_list, target_label):
    for value in cols_list:
        if value == target_label:
            cols_list.remove(value)
            break
    return cols_list


# the main function to run the pre-processing phase and to fit the model to training data #
def run_model(df):

    # split data-time column if exists
    cd.split_datetime_col(df)
    
    # classify columns to categorical and numerical
    summarized_df, numeric_cols, nominal_cols, ordinal_cols = col_classify.get_numeric_nominal_ordinal_cols(df)
    numeric_cols = drop_target_label(numeric_cols, target_label)
    nominal_cols = drop_target_label(nominal_cols, target_label)
    ordinal_cols = drop_target_label(ordinal_cols, target_label)

    simple_imputer = SimpleImputer(strategy="most_frequent", copy=False)

    numeric_transformer = Pipeline(
        steps=
        [
            ('imputer', SimpleImputer(missing_values=np.NaN, strategy='mean', copy=False)),
            ('scaler', StandardScaler())
        ]
    )

    categorical_transformer_ordinal = Pipeline(
        steps=
        [
            ('imputer', simple_imputer),
            ('LabelEncoder', ce.BinaryEncoder())
        ]
    )

    categorical_transformer_nominal = Pipeline(
        steps=
        [
            ('imputer', simple_imputer),
            ('OneHotEncoder', ce.OneHotEncoder())
        ]
    )

    # combine transformers in one Preprocessor 
    preprocessor = ColumnTransformer(
        transformers=
        [
            ('num', numeric_transformer, numeric_cols),
            ('cat_ordinal', categorical_transformer_ordinal, ordinal_cols),
            ('cat_nominal', categorical_transformer_nominal, nominal_cols)
        ]
    )
    
    # append classifier to pre-processing pipeline.
    classifier = Pipeline(
        steps=
        [
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))
        ]
    )
    
    # Features
    X = df.drop(target_label, axis=1)

    # Target Label
    y = df[target_label]
    
    # Split Data to Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Fit Model
    classifier.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)
    return score


main_func()
