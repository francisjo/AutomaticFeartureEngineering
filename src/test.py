import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from decimal import Decimal
'''
crosstab = pd.crosstab(df['Cabin'], df["Fare"])
# crosstab = df.Name.groupby(df['Name']).count()
stat, p, dof, expected = chi2_contingency(crosstab)

prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
    print('Dependent')
else:
    print('Independent')
'''


'''
def columns_correlation_chi2(df):
    result_isdependent = {}
    result_critical = {}
    result_stat = {}
    result_dof = {}
    for col1 in df.columns:
        chi2_values = pd.DataFrame()
        for col2 in df.columns:
            if col1 != col2:
                crosstab = pd.crosstab(df[col1], df[col2])
                stat, p, dof, expected = chi2_contingency(crosstab)
                prob = 0.95
                critical = chi2.ppf(prob, dof)
                if abs(stat) >= critical:
                    chi2_values[col2] = [1, critical, stat, dof]
                else:
                    chi2_values[col2] = [0, critical, stat, dof]
        chi2_values_T = chi2_values.T
        dependent_min = chi2_values_T[chi2_values_T[0] == 1].groupby(chi2_values_T[0]).min()
        if len(dependent_min) != 0:
            print(type(dependent_min))
            result_isdependent[col1] = dependent_min[0]
            result_critical[col1] = dependent_min[1]
            result_stat[col1] = dependent_min[2]
            result_dof[col1] = dependent_min[3]
        else:
            print(type(dependent_min))
            independent_min = chi2_values_T[chi2_values_T[0] == 0].groupby(chi2_values_T[0]).min()
            result_isdependent[col1] = independent_min[0]
            result_critical[col1] = independent_min[1]
            result_stat[col1] = independent_min[2]
            result_dof[col1] = independent_min[3]

    return result_isdependent, result_critical, result_stat, result_dof


dff = df[['Name', 'Sex', 'Age']].copy()
result_isdependent, result_critical, result_stat, result_dof = columns_correlation_chi2(dff)
x=1
'''
'''
def columns_correlation_spearmanr(df):
    result = {}
    for col1 in df.columns:
        spearmanr_values = pd.Series()
        for col2 in df.columns:
            if col1 != col2:
                value = stats.spearmanr(df[col1], df[col2])[0]
                spearmanr_values[col2] = value
        max_val = spearmanr_values.max()
        min_val = spearmanr_values.min()
        max_val_abs = [max_val, abs(max_val)]
        min_val_abs = [min_val, abs(min_val)]
        strong_value = max(max_val_abs[1], min_val_abs[1])
        if strong_value == max_val_abs[1]:
            strong_value = max_val_abs[0]
        else:
            strong_value = min_val_abs[0]
        result[col1] = round(Decimal(strong_value), 3)
    return result


result = columns_correlation_spearmanr(dff)

print(result)
'''



# ========================  OLD MAIN FUNCTION ========================== #

'''

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
    #cd.split_datetime_col(df)

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

'''







#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Wed Jun 19 22:33:54 2019

@author: basha
"""



import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, Imputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression



adult = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/train.csv'
df = pd.read_csv(adult)

#------------------ columns lists ------------------------------
# numerical
numeric_list = ['Age', 'SibSp', 'Parch', 'Fare']

# categorical

single_col = 'PassengerId'
other_cols = ['Pclass', 'Sex',
            'Ticket', 'Cabin',
            'Embarked', 'Name']

numeric_transformer = Pipeline(
        steps=
        [
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean', copy=False)),
            ('scaler', StandardScaler())
        ]
    )
single_col_transformer = Pipeline(
            steps=
            [
                ('imputer', SimpleImputer(missing_values=np.nan, strategy="most_frequent", copy=False)),
                ("key1", ce.TargetEncoder())
            ]
        )

other_cols_transformer = Pipeline(
            steps=
            [
                ('imputer', SimpleImputer(missing_values=np.nan, strategy="most_frequent", copy=False)),
                ("key2", ce.TargetEncoder())
            ]
        )

preprocessor = ColumnTransformer(
        transformers=
        [
            ('single_col', single_col_transformer, [single_col]),
            ('num', numeric_transformer, numeric_list),
            ('other_cols', other_cols_transformer, other_cols)
        ]
    )
clf = Pipeline(
    steps=
    [
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100000))
    ]
)
y = df['Survived']
X = df.drop('Survived', axis=1)
#X = df[['PassengerId', 'Age','Pclass','Sex','Embarked']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf.fit(X_train, y_train)
#clf.fit(df[['Name']], y)

#----------------------- Model evaluation ----------------------------------
probs_train = clf.predict_proba(X_train)[:, 1]
probs_test = clf.predict_proba(X_test)[:, 1]
print("score train: {}".format(roc_auc_score(y_train, probs_train)))
print("score test: {}".format(roc_auc_score(y_test, probs_test)))
#------------------------------------------------------------------------------







































