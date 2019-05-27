import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

import column_type_classification as col_classify

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# load dataset into a pandas data-frame
def load_data_online():
    titanic = 'https://raw.githubusercontent.com/nphardly/titanic/master/data/inputs/train.csv'
    car = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/car.csv'
    adult = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/adult.csv'
    heart = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/heart.csv'
    bridges = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/bridges.csv'

    df_titanic = pd.read_csv(titanic)
    df_car = pd.read_csv(car)
    df_adult = pd.read_csv(adult)
    df_heart = pd.read_csv(heart)
    df_bridges = pd.read_csv(bridges)
    df_dict = {"titanic": df_titanic, "car": df_car, "adult": df_adult} #, "heart": df_heart

    return df_dict


def load_data_local():
    titanic = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\train.csv'
    car = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\car.csv'
    adult = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\adult.csv'
    heart = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\heart.csv'
    bridges = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\bridges.csv'
    df_titanic = pd.read_csv(titanic)
    df_car = pd.read_csv(car)
    df_adult = pd.read_csv(adult)
    df_heart = pd.read_csv(heart)
    df_bridges = pd.read_csv(bridges)
    df_dict = {"titanic": df_titanic, "car": df_car, "adult": df_adult} #, "heart": df_heart
    return df_dict


def run_model_tree(df):
    # Features
    X = df.drop("Cls-Result", axis=1)

    # Target Label
    y = df["Cls-Result"]

    # Split Data to Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)
    # Fit Model
    clf = tree.DecisionTreeClassifier(criterion="gini", random_state=3, max_depth=4)
    clf.fit(X_train, y_train)

    # ==== Test Model on new Data [bridges.csv] ====
    bridges = 'https://raw.githubusercontent.com/francisjo/AutomaticFeartureEngineering/master/Datasets/bridges.csv'
    # bridges = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\bridges.csv'
    bridges_df = pd.read_csv(bridges)
    bridges_dict = {"bridges": bridges_df}
    summarized_test = col_classify.get_summarized_df(bridges_dict)
    summarized_test = summarized_test.set_index("index")
    X_test = summarized_test.drop("Cls-Result", axis=1)
    y_test = summarized_test["Cls-Result"]

    # ================================

    y_predict = clf.predict(X_test)
    print("------Classification Report-------")
    print(classification_report(y_test, y_predict))
    score = clf.score(X_test, y_test)
    print("Decision Tree Score: ", score)
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_predict)
    plot_confusion_matrix(cm, ["Numerical", "Nominal", "Ordinal"])
    print("--- Prediction Results ---")
    print(y_predict)
    print("--- Ground-truth ---")
    print(y_test)
    print("Features Importance :  ", clf.feature_importances_)

    # Applying K-Fold Cross Validation
    print("------K-Fold Cross Validation-------")
    accuracies = cross_val_score(estimator=clf, X=X_test, y=y_test, cv=5)
    print("Cross-Validation Accuracies:  ", accuracies)
    print("Cross-Validation Mean Accuracy =  ", accuracies.mean())

    # ======== Plot Decision Tree ===========
    plot_decision_tree_model(clf, X)


def plot_confusion_matrix(cm, target_names, normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Oranges')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix', fontdict={'family': 'arial', 'weight': 'bold', 'size': 14})
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    font = {'family': 'arial',
            'color': 'black',
            'weight': 'normal',
            'size': 12,
            }

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black", fontdict=font)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black", fontdict=font)

    plt.tight_layout()
    plt.ylabel('True Labels', fontdict=font)
    plt.xlabel('Predicted Labels\n\nAccuracy={:0.3f}; Misclass={:0.3f}'.format(accuracy, misclass), fontdict=font)
    plt.show()


def plot_decision_tree_model(clf, X):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    feature_names=X.columns,
                    class_names=True,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree.png')
    Image(graph.create_png())


# passing the data-frame to the run_model() function
def main_func():
    df_dict = load_data_online()
    # df_dict = load_data_local()
    summarized_dfs = col_classify.get_summarized_df(df_dict)
    summarized_dfs = summarized_dfs.set_index("index")
    run_model_tree(summarized_dfs)


main_func()


