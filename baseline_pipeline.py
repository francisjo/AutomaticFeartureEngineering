import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
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
    clf = tree.DecisionTreeClassifier(criterion="gini", random_state=3)
    clf.fit(X_train, y_train)

    # ==== Test Model on new Data [bridges.csv] ====
    car1 = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car1.csv'
    # bridges = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\bridges.csv'
    car1_df = pd.read_csv(car1)
    car1_dict = {"car1": car1_df}
    summarized_test = col_classify.get_summarized_df(car1_dict)
    summarized_test = summarized_test.set_index("index")
    X_test = summarized_test.drop("Cls-Result", axis=1)
    y_test = summarized_test["Cls-Result"]

    y_predict = clf.predict(X_test)
    print(y_predict)
    print("------Classification Report-------")
    print(classification_report(y_test, y_predict))
    score = clf.score(X_test, y_test)
    print("Decision Tree Score: ", score)

    print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    gp.plot_confusion_matrix(cm, ["Numerical", "Nominal", "Ordinal"])

    # Plot Decision Tree
    gp.plot_decision_tree_model(clf, X)

    print("Features Importance:  ", clf.feature_importances_)
    '''
    # Applying K-Fold Cross Validation
    print("------K-Fold Cross Validation-------")
    accuracies = cross_val_score(estimator=clf, X=X_test, y=y_test, cv=5)
    print("Cross-Validation Accuracies:  ", accuracies)
    print("Cross-Validation Mean Accuracy =  ", accuracies.mean())
    '''
    predictdf = pd.DataFrame(y_predict)
    predictdf.columns = ['new_clf']

    print("--- Ground-truth and Prediction Results ---")
    final_clf_df = pd.concat([y_test.reset_index(), predictdf], axis=1)
    print(final_clf_df)
    return final_clf_df


def main_func():
    w2v.init()
    df_dict = ld.load_data_online()
    # df_dict = ld.load_data_local()
    summarized_dfs = col_classify.get_summarized_df(df_dict)
    summarized_dfs = summarized_dfs.set_index("index")
    final_clf_df = run_model_tree(summarized_dfs)
    car1 = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car1.csv'
    car1_df = pd.read_csv(car1)
    enc_meth.run_model_representation(car1_df, final_clf_df)


main_func()

