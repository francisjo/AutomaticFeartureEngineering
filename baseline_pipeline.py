import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree

import column_type_classification as col_classify


# load dataset into a pandas data-frame
def load_data_online():
    titanic = 'https://raw.githubusercontent.com/nphardly/titanic/master/data/inputs/train.csv'
    car = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car.csv'
    adult = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/adult.csv'
    heart = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/heart.csv'
    df_titanic = pd.read_csv(titanic)
    df_car = pd.read_csv(car)
    df_adult = pd.read_csv(adult)
    df_heart = pd.read_csv(heart)
    df_dict = {"titanic": df_titanic, "car": df_car, "adult": df_adult, "heart": df_heart}
    return df_dict


def load_data_local():
    titanic = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\train.csv'
    car = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\car.csv'
    adult = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\adult.csv'
    heart = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\adult.csv'
    df_titanic = pd.read_csv(titanic)
    df_car = pd.read_csv(car)
    df_adult = pd.read_csv(adult)
    df_heart = pd.read_csv(heart)
    df_dict = {"titanic": df_titanic, "car": df_car, "adult": df_adult, "heart": df_heart}
    return df_dict


def run_model_tree(df):
    # Features
    X = df.drop("Cls-Result", axis=1)

    # Target Label
    y = df["Cls-Result"]

    # Split Data to Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # test= run_model1(df_heart_dict).set_index("index")
    # X_test = test.drop("Cls-Result", axis=1)
    # y_test = test["Cls-Result"]

    # Fit Model
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print("Decision Tree Score: ", score)
    print("------Prediction Result-------")
    print(predict)
    print("------Old Correct Labels-------")
    print(y_test)


# passing the data-frame to the run_model() function
def main_func():
    df_dict = load_data_online()
    # df_dict = load_data_local()
    summarized_dfs = col_classify.get_summarized_df(df_dict)
    summarized_dfs = summarized_dfs.set_index("index")
    run_model_tree(summarized_dfs)


main_func()


