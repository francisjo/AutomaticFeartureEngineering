import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV , LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

import word2vec_load as w2v
import column_type_classification as col_classify
#import generate_plots as gp
import load_datasets as ld

def train_test_split_by_df(df,df_name):
    train = df[df['Df_Name'] != df_name]
    test = df[df['Df_Name'] == df_name]
    X_train = train.drop(["Cls-Result", "Df_Name"], axis=1)
    y_train = train["Cls-Result"]
    X_test = test.drop(["Cls-Result", "Df_Name"], axis=1)
    y_test = test["Cls-Result"]
    return X_train, X_test, y_train, y_test
'''
def conf_matrix_method_and_plots(y_test, y_predict,clf,X):
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    gp.plot_confusion_matrix(cm, ["Numerical", "Nominal", "Ordinal"])
    # Plot Decision Tree
    gp.plot_decision_tree_model(clf, X)
'''
def run_model_tree(df):
    df_names = df['Df_Name'].unique()
    final_clf_dfs = pd.DataFrame()
    final_clf_importance_score_dfs = pd.DataFrame()
    for df_name in df_names:
        # Split Data to Train and Test Data
        X_train, X_test, y_train, y_test = train_test_split_by_df(df, df_name)
        '''
        # Features
        X = df.drop("Cls-Result", axis=1)
        # Target Label
        y = df["Cls-Result"]
        # Split Data to Train and Test Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=42
        '''
        # Fit Model
        # clf = tree.DecisionTreeClassifier(random_state=3)
        # grid_search_method(clf, X_train, y_train)
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=4, max_features=None, min_samples_leaf=2, min_samples_split= 2, splitter="random", random_state=3)
        clf.fit(X_train, y_train)
        '''
        # ==== Test Model on new Data [bridges.csv] ====
        titanic = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/train.csv'
        titanic_df = pd.read_csv(titanic)
        titanic_dict = {"titanic": titanic_df}
        
        summarized_test = col_classify.get_summarized_df(titanic_dict)
        summarized_test = summarized_test.set_index("index")
        X_test = summarized_test.drop("Cls-Result", axis=1)
        y_test = summarized_test["Cls-Result"]
    '''
        print("***************************"+df_name+"**********************************")
        y_predict = clf.predict(X_test)
        print(y_predict)
    
        ####Matrix_confusion_and_plots
        #conf_matrix_method_and_plots(y_test, y_predict, clf, df.drop(["Cls-Result", "Df_Name"], axis=1))
        feature_importances = clf.feature_importances_

        print("Features Importance:  ", feature_importances)
    
        # print("------Classification Report-------")
        # print(classification_report(y_test, y_predict))
        score = clf.score(X_test, y_test)
        print("Decision Tree Score: ", score)
        predictdf = pd.DataFrame(y_predict)
        predictdf.columns = ['new_clf']
        print("--- Ground-truth and Prediction Results ---")
        #feature_importancesseries = pd.Series(feature_importances)
        final_clf_importance_score_df = pd.DataFrame([feature_importances],columns=X_test.columns.tolist())
        final_clf_importance_score_df["Score"] = score
        final_clf_importance_score_df["Df_name"] = df_name
        final_clf_importance_score_dfs = pd.concat([final_clf_importance_score_dfs,final_clf_importance_score_df])
        final_clf_df = pd.concat([y_test.reset_index(), predictdf], axis=1)
        final_clf_df['Df_Name'] = df_name
        print(final_clf_df)
        final_clf_dfs = pd.concat([final_clf_dfs, final_clf_df])
    final_clf_importance_score_dfs.to_csv('final_clf_importance_score_dfs_withskew.csv', sep=',', header=True)
    final_clf_dfs.to_csv('final_clf_dfs_withskew.csv', sep=',', header=True)
    return final_clf_df



def main_func():
    w2v.init()
    df_dict = ld.load_data_online()
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
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
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
    features_importance = gd_sr.best_estimator_.feature_importance()

    print("best_parameters: ", best_parameters)
    print("Best score: ", best_result)
    print("Features Importance:  ", features_importance)


def cross_validation_method(clf, X_test, y_test):
    LeaveOneOut()
    # Applying K-Fold Cross Validation
    print("------K-Fold Cross Validation-------")
    accuracies = cross_val_score(estimator=clf, X=X_test, y=y_test, cv=5)
    print("Cross-Validation Accuracies:  ", accuracies)
    print("Cross-Validation Mean Accuracy =  ", accuracies.mean())


main_func()

