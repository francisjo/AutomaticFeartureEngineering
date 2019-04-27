# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas.api.types as ptypes
import category_encoders as ce

df = pd.read_csv('https://raw.githubusercontent.com/nphardly/titanic/master/data/inputs/train.csv')
#df = pd.read_csv('https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/heart.csv')
#df = pd.read_csv('https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car_data.csv')
#df = pd.read_csv('https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/smallwikipedia.csv',delimiter=';')
numericList = []
categoricalList = []
threshold = 0.20
targetLabel = 'Survived'
### Replace Null values to NaN
df.fillna(np.NaN,inplace=True)
##convert Date col to Datetime and split it to Year,Month,Day Columns###
#df['Date'] = pd.to_datetime(df['Date'])
def splitDatecol():
    
    for col in df.columns:  
        if(ptypes.is_datetime64_ns_dtype(df[col])):
            df[col+'_Day'] = pd.DatetimeIndex(df[col]).day
            df[col+'_Month'] = pd.DatetimeIndex(df[col]).month
            df[col+'_Year'] = pd.DatetimeIndex(df[col]).year
         
##### Classify Columns Wih Threshold####################
def classifyColumnsWithThreshold(threshold=0.2):
    for col in df.columns:
        if (col == targetLabel):
            continue
        elif (ptypes.is_string_dtype(df[col])):
            categoricalList.append(col)
        elif (ptypes.is_numeric_dtype(df[col])):
            col_count = len(df[col].unique())
            percent = col_count / df[col].count()
            if (percent < threshold):
                categoricalList.append(col)
            else:
                numericList.append(col)
splitDatecol()             
classifyColumnsWithThreshold(threshold)
################################################
def categorical_transformermethod(categoricalmethod1):
    
    if (categoricalmethod1 =='BinaryEncoder'):
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy="most_frequent",copy=False))
        ,('BinaryEncoder', ce.BinaryEncoder(handle_unknown='impute'))
        ])
    elif (categoricalmethod1 =='OneHotEncoder'):
         categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent",copy=False))
        ,('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
    return categorical_transformer
#################################################
    
def run_modul(categoricalmethod='OneHotEncoder'):
 # We create the preprocessing pipelines for both numeric and categorical data.
        # 1- For Numeric data
        numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.NaN, strategy='mean',copy=False)),
        ('scaler', StandardScaler())])
        # 2 -For categorical data
        categorical_transformer=categorical_transformermethod(categoricalmethod)
        # 3- Combine Transformers in one Preprocessor 
        preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_transformer, numericList),
        ('cat', categorical_transformer, categoricalList)])
        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        clf = Pipeline(steps=[('preprocessor', preprocessor),
          ('classifier', LogisticRegression(solver='lbfgs'))])
        ##### Features
        X = df.drop(targetLabel, axis=1)
        ##### Target Label
        y = df[targetLabel]
        #### Split Data to Train and Test Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
        # Fit Modul
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)
#########################################################################
        
classicalencoder = ['OneHotEncoder','BinaryEncoder']
for name in classicalencoder:
    print(name)
    result = run_modul(name)
    print("model score with" + name + ":%.3f" %  result)
