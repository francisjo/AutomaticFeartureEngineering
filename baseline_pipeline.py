# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

#df = pd.read_csv('https://raw.githubusercontent.com/nphardly/titanic/master/data/inputs/train.csv')
#df = pd.read_csv('Desktop/Data Science Applications Project/AutomaticFeartureEngineering/heart.csv')
df = pd.read_csv('Desktop/Data Science Applications Project/AutomaticFeartureEngineering/car_data.csv')

numericList = []
categoricalList = []
threshold = 0.20
targetLabel = 'price'


def classifyColumnsWithThreshold(threshold=0.2):
    for col in df.columns:
        if (col == targetLabel):
            continue
        elif (is_string_dtype(df[col])):
            categoricalList.append(col)
        elif (is_numeric_dtype(df[col])):
            col_count = len(df[col].unique())
            percent = col_count / df[col].count()
            if (percent < threshold):
                categoricalList.append(col)
            else:
                numericList.append(col)
            
classifyColumnsWithThreshold(threshold)

numeric_features = numericList
categorical_features = categoricalList

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    #('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

X = df.drop(targetLabel, axis=1)
y = df[targetLabel]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
