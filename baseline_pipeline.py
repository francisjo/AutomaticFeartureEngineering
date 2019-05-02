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

##### fill null values with respect to the column data-type ######
def fill_null_data(col, col_type):
    if (df[col].isnull().sum() > 0 and col_type == 'string'):
        df[col].fillna('missing', inplace=True)
    elif (df[col].isnull().sum() > 0 and col_type == 'numeric'):
        df[col].fillna(df[col].mean(), inplace=True)
            
#####convert Date col to Datetime and split it to Year,Month,Day Columns###
def split_datetime_col():
    #df['Date'] = pd.to_datetime(df['Date'])
    for col in df.columns:  
        if(ptypes.is_datetime64_ns_dtype(df[col])):
            df[col+'_Day'] = pd.DatetimeIndex(df[col]).day
            df[col+'_Month'] = pd.DatetimeIndex(df[col]).month
            df[col+'_Year'] = pd.DatetimeIndex(df[col]).year
         
##### Detect categorical and numrical variables by Frequency of unique value in Columns Wih Threshold####################
def classify_columns_statistically(threshold=0.2):
    numeric_list = []
    categoric_list = []
    for col in df.columns:
        if (col == targetLabel):
            continue
        elif (ptypes.is_string_dtype(df[col])):
            fill_null_data(col, 'string')
            categoric_list.append(col)
        elif (ptypes.is_numeric_dtype(df[col])):
            fill_null_data(col, 'numeric')
            col_count = len(df[col].unique())
            percent = col_count / df[col].count()
            if (percent < threshold):
                categoric_list.append(col)
            else:
                numeric_list.append(col)
    return numeric_list, categoric_list

##### Detect categorical and numrical variables by Frequency of unique value in Columns Wih Threshold####################
def classify_columns_distribution(threshold=0.2):
    numeric_list = []
    categoric_list = []
    for col in df.columns:
        if (col == targetLabel):
            continue
        elif (ptypes.is_string_dtype(df[col])):
            fill_null_data(col, 'string')
            categoric_list.append(col)
        elif (ptypes.is_numeric_dtype(df[col])):
            fill_null_data(col, 'numeric')
            value_count = df.groupby(col)[col].count()
            col_dist = value_count /df[col].count()
            if((col_dist[col_dist > threshold].count())>= 1):
                categoric_list.append(col)
            else:
                numeric_list.append(col)
    print('categoric_list :')
    print(categoric_list)
    print('numeric_list :')
    print(numeric_list)
    numeric_list1,categoric_list1 = classify_columns_correlation(numeric_list, categoric_list)
    print('categoric_list1 :')
    print(categoric_list1)
    print('numeric_list1 :')
    print(numeric_list1)
    return numeric_list1, categoric_list1

def classify_columns_correlation(numeric_list, categoric_list,threshold=0.2):
    corr = df.corr()
    for value in categoric_list:
        if(ptypes.is_string_dtype(df[value])):
            print('String value: ' + value)
            continue
        corr_values = corr[value]
        corr_values = corr_values.drop(value)
        for i in corr_values:
            corr_values
            if (i > 0.5 or i < -0.5) :
                print('value: ' + value)
                categoric_list.remove(value)
                numeric_list.append(value)
                break
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
    numerical_features, categorical_features = classify_columns_distribution(threshold)
    
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
    #score = classifier.score(X_test, y_test)
    return classifier
       
#classical_encoders = ['OneHotEncoder','BinaryEncoder', 'HashingEncoder']
classical_encoders = ['HashingEncoder']
for encoder in classical_encoders:
   # result = run_model(encoder)
   classifier = run_model(encoder)
   coef= classifier.named_steps['classifier'].coef_
   len(coef)
  # print("Model Score with: " + encoder + " : %.3f" %  result)
   # print('--------------')
