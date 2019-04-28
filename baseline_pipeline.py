# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas.api.types as ptypes
import category_encoders as ce

titanic = 'https://raw.githubusercontent.com/nphardly/titanic/master/data/inputs/train.csv'
heart = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/heart.csv'
car = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car_data.csv'
wikipedia = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/smallwikipedia.csv'

df = pd.read_csv(titanic)
#df = pd.read_csv(heart)
#df = pd.read_csv(car)
#df = pd.read_csv(wikipedia, delimiter=';')  #for datetime column

## global variables
threshold = 0.20
targetLabel = 'Survived'

### Replace Null values to NaN
df.fillna(np.NaN,inplace=True)

#####convert Date col to Datetime and split it to Year,Month,Day Columns###
def split_datetime_col():
    #df['Date'] = pd.to_datetime(df['Date'])
    for col in df.columns:  
        if(ptypes.is_datetime64_ns_dtype(df[col])):
            df[col+'_Day'] = pd.DatetimeIndex(df[col]).day
            df[col+'_Month'] = pd.DatetimeIndex(df[col]).month
            df[col+'_Year'] = pd.DatetimeIndex(df[col]).year
         
##### Classify Columns Wih Threshold####################
def classify_columns_statistically(threshold=0.2):
    numeric_list = []
    categoric_list = []
    for col in df.columns:
        if (col == targetLabel):
            continue
        elif (ptypes.is_string_dtype(df[col])):
            categoric_list.append(col)
        elif (ptypes.is_numeric_dtype(df[col])):
            col_count = len(df[col].unique())
            percent = col_count / df[col].count()
            if (percent < threshold):
                categoric_list.append(col)
            else:
                numeric_list.append(col)
    return numeric_list, categoric_list

##### Transform the categorical data by different encoding methods ###########
def categorical_transformer_method(categorical_method):
    simple_imputer = SimpleImputer(strategy="most_frequent",copy=False)
    if (categorical_method =='BinaryEncoder'):
        categorical_transformer = Pipeline(
                steps=[
                        ('imputer', simple_imputer),
                        ('BinaryEncoder', ce.BinaryEncoder())
                ])
    elif (categorical_method =='OneHotEncoder'):
        categorical_transformer = Pipeline(
                steps=[
                        ('imputer', simple_imputer),
                        ('onehot', ce.OneHotEncoder())
                ])
    elif (categorical_method =='HashingEncoder'):
        categorical_transformer = Pipeline(
                steps=[
                        ('imputer', simple_imputer),
                        ('onehot', ce.HashingEncoder())
                ])
    return categorical_transformer

##### the main function to run the preprocessing phase and to fit the model to training data #######
def run_model(categorical_method):
    
    # split datatime column if exists
    split_datetime_col()
    
    # classify columns to categorical and numerical
    numerical_features, categorical_features = classify_columns_statistically(threshold)
    
    # impute and scaling numerical features
    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.NaN, strategy='mean',copy=False)),
    ('scaler', StandardScaler())])
    
    # try many different categorical data encoders
    categorical_transformer = categorical_transformer_method(categorical_method)
    
    # combine transformers in one Preprocessor 
    preprocessor = ColumnTransformer(
            transformers=[
                    ('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                    ]
            )
    
    # append classifier to preprocessing pipeline.
    classifier = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))
            ])
    
    ##### Features
    X = df.drop(targetLabel, axis=1)
    
    ##### Target Label
    y = df[targetLabel]
    
    #### Split Data to Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    # Fit Model
    classifier.fit(X_train, y_train)
    return classifier.score(X_test, y_test)
        

classical_encoders = ['OneHotEncoder','BinaryEncoder', 'HashingEncoder']
for encoder in classical_encoders:
    result = run_model(encoder)
    print("Model Score with: " + encoder + " : %.3f" %  result)
    print('--------------')
