import pandas as pd
import numpy as np
import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO
import graphviz
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import urllib as urllo
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
import column_type_classification as col_classify
import data_cleaning as cd
import category_encoders as ce


# global variables
threshold = 0.20
target_label = 'price'
heart = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\heart.csv'
df_heart = pd.read_csv(heart)
df_heart_dict={"heart" : df_heart}
# load dataset into a pandas data-frame
def load_data():

    #titanic = 'https://raw.githubusercontent.com/nphardly/titanic/master/data/inputs/train.csv'
    #heart = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/heart.csv'
    #car = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car_data.csv'
    titanic = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\train.csv'
    car = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\car_data.csv'
    wikipedia = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/smallwikipedia.csv'
    adult = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\phpMawTba.csv'
    df_titanic = pd.read_csv(titanic)
    df_car = pd.read_csv(car)
    df_adult = pd.read_csv(adult)
    df_dict = {"titanic": df_titanic, "car": df_car, "adult": df_adult, "heart": df_heart}
    return df_dict


# passing the data-frame to the run_model() function
def main_func():
    df_dict = load_data()
    df = pd.DataFrame()
    result = run_model1(df_dict)
    result = result.set_index("index")
    run_model_tree(result)

def run_model_tree(df):
    # Features
    X = df.drop("Cls-Result", axis=1)

    # Target Label
    y = df["Cls-Result"]

    # Split Data to Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
  #  test= run_model1(df_heart_dict).set_index("index")
   # X_test = test.drop("Cls-Result", axis=1)
   # y_test = test["Cls-Result"]
    # Fit Model
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    perdict = clf.predict(X_test)
    perdict = perdict.T
    score = clf.score(X_test, y_test)
    #result_y["New_Result"] = perdict
    print(perdict)
    print("-------------")
    print(y_test)
    print("-------------")
    #dot_data = StringIO()
    '''
   tree.export_graphviz(clf, out_file=dot_data,
                                    filled=True, rounded=True,
                                    special_characters=True)
    #graph = graphviz.Source(dot_data)
    #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #Image(graph.create_png())
   # graph.write_png('tree.png')
   '''
    x = 1

def run_model1(df_dict):
    summarized_dfs = col_classify.get_summarized_df(df_dict)
    x = 0
   # summarized_df_heart = col_classify.get_summarized_df(df_dict)
   # summarized_df_df_car = col_classify.get_summarized_df(df_dict)
    return summarized_dfs
# drop the target label from a list
def drop_target_label(cols_list, target_label):
    for value in cols_list:
        if value == target_label:
            cols_list.remove(value)
            break
    return cols_list

'''
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

main_func()
