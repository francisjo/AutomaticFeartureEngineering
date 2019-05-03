import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from columns_classification import ClassifyColumns as cc
from data_cleaning import CleaningData as cd
from data_preprocessing import PreprocessingData as predata

titanic = 'https://raw.githubusercontent.com/nphardly/titanic/master/data/inputs/train.csv'
heart = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/heart.csv'
car = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car_data.csv'
wikipedia = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/smallwikipedia.csv'

df = pd.read_csv(titanic)
# df = pd.read_csv(heart)
# df = pd.read_csv(car)
# df = pd.read_csv(wikipedia, delimiter=';')  #for datetime column

# global variables
threshold = 0.20
targetLabel = 'Survived'

# the main function to run the preprocessing phase and to fit the model to training data #
def run_model(categorical_method):
    
    # split datatime column if exists
    cd.split_datetime_col(df)
    
    # classify columns to categorical and numerical
    numerical_features, categorical_features = cc.columns_distribution_classification(df, targetLabel)

    # impute and scaling numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.NaN, strategy='mean', copy=False)),
        ('scaler', StandardScaler())])
    
    # try many different categorical data encoders
    categorical_transformer = predata.categorical_transformer_method(categorical_method)
    
    # combine transformers in one Preprocessor 
    preprocessor = ColumnTransformer(
        transformers=
        [
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # append classifier to preprocessing pipeline.
    classifier = Pipeline(
        steps=
        [
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))
        ]
    )
    
    # Features
    X = df.drop(targetLabel, axis=1)
    
    # Target Label
    y = df[targetLabel]
    
    # Split Data to Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    # Fit Model
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    return score


classical_encoders = ['OneHotEncoder', 'BinaryEncoder', 'HashingEncoder']
for encoder in classical_encoders:
    result = run_model(encoder)
    print("Model Score with: " + encoder + " : %.3f" % result)
    print('--------------')