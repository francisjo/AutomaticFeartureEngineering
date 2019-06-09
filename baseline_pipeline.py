import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

import word2vec_load as w2v
import column_type_classification as col_classify
import encoders_methods as enc_meth
import generate_plots as gp
import load_datasets as ld


def run_model_tree(df):
    # Features
    X = df.drop("Cls-Result", axis=1)

    # Target Label
    y = df["Cls-Result"]

    # Split Data to Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)
    # Fit Model
    clf = tree.DecisionTreeClassifier(random_state=3)
    grid_search_method(clf, X_train, y_train)
    '''
    clf.fit(X_train, y_train)

    # ==== Test Model on new Data [bridges.csv] ====
    titanic = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/train.csv'
    titanic_df = pd.read_csv(titanic)
    titanic_dict = {"titanic": titanic_df}
    summarized_test = col_classify.get_summarized_df(titanic_dict)
    summarized_test = summarized_test.set_index("index")
    X_test = summarized_test.drop("Cls-Result", axis=1)
    y_test = summarized_test["Cls-Result"]

    y_predict = clf.predict(X_test)
    print(y_predict)

    print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    gp.plot_confusion_matrix(cm, ["Numerical", "Nominal", "Ordinal"])

    # Plot Decision Tree
    gp.plot_decision_tree_model(clf, X)

    print("Features Importance:  ", clf.feature_importances_)

    # print("------Classification Report-------")
    # print(classification_report(y_test, y_predict))
    score = clf.score(X_test, y_test)
    print("Decision Tree Score: ", score)





    #predictdf = pd.DataFrame(y_predict)
    #predictdf.columns = ['new_clf']

    print("--- Ground-truth and Prediction Results ---")
    final_clf_df = pd.concat([y_test.reset_index(), predictdf], axis=1)
    print(final_clf_df)
    return final_clf_df
    '''


def main_func():
    w2v.init()
    df_dict = ld.load_data_local()
    # df_dict = ld.load_data_local()
    summarized_dfs = col_classify.get_summarized_df(df_dict)
    summarized_dfs = summarized_dfs.set_index("index")
    # final_clf_df = run_model_tree(summarized_dfs)
    run_model_tree(summarized_dfs)
    # adult = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/adult.csv'
    # adult_df = pd.read_csv(adult)
    # enc_meth.run_model_representation(adult_df, final_clf_df)


def grid_search_method(classifier, X_train, y_train):
    grid_param = {
        'n_estimators': [100, 300, 500, 800, 1000],
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'bootstrap': [True, False],
        'max_depth ': [3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6],
        'max_features': [None, "auto", "sqrt", "log2"]
    }
    gd_sr = GridSearchCV(estimator=classifier,
                         param_grid=grid_param,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=-1)
    gd_sr.fit(X_train, y_train)
    best_parameters = gd_sr.best_params_
    best_result = gd_sr.best_score_
    print("best_parameters: ", best_parameters)
    print("Best score: ", best_result)


def cross_validation_method(clf, X_test, y_test):
    # Applying K-Fold Cross Validation
    print("------K-Fold Cross Validation-------")
    accuracies = cross_val_score(estimator=clf, X=X_test, y=y_test, cv=5)
    print("Cross-Validation Accuracies:  ", accuracies)
    print("Cross-Validation Mean Accuracy =  ", accuracies.mean())


main_func()

