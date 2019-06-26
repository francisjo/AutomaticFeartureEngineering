from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# drop the target label from a list
def drop_target_label(cols_list, target_label):
    for value in cols_list:
        if value == target_label:
            cols_list.remove(value)
            break
    return cols_list


def get_cols_type_lists(clf_df, target_label):
    # classify columns to categorical and numerical
    numeric_cols = drop_target_label(clf_df.loc[clf_df["new_clf"] == 1, "index"].tolist(), target_label)
    nominal_cols = drop_target_label(clf_df.loc[clf_df["new_clf"] == 2, "index"].tolist(), target_label)
    ordinal_cols = drop_target_label(clf_df.loc[clf_df["new_clf"] == 3, "index"].tolist(), target_label)
    return numeric_cols, nominal_cols, ordinal_cols


def get_classifier(key1, key2, encoder1, encoder2, clf_df, target_label):
    numeric_cols, nominal_cols, ordinal_cols = get_cols_type_lists(clf_df, target_label)
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
            ('imputer', SimpleImputer(strategy="most_frequent", copy=False)),
            (key1, encoder1)
        ]
    )
    categorical_transformer_nominal = Pipeline(
        steps=
        [
            ('imputer', SimpleImputer(strategy="most_frequent", copy=False)),
            (key2, encoder2)
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=
        [
            ('num', numeric_transformer, numeric_cols),
            ('cat_ordinal', categorical_transformer_ordinal, ordinal_cols),
            ('cat_nominal', categorical_transformer_nominal, nominal_cols)
        ]
    )
    classifier = Pipeline(
        steps=
        [
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=100000))
        ]
    )
    return classifier


def preprocessing_pipeline(clf_df, target_label):
    encoder_dict = {#'OneHotEncoder': ce.OneHotEncoder(),
                    #'BinaryEncoder': ce.BinaryEncoder(),
                    #'HashingEncoder': ce.HashingEncoder(),
                    #'LabelEncoder': MultiColumnLabelEncoder(),
                    # 'OrdinalEncoder': ce.OrdinalEncoder(),
                    # 'PolynomialEncoder': ce.PolynomialEncoder(),
                    # 'TargetEncoder': ce.TargetEncoder(),
                    # 'HelmertEncoder': ce.HelmertEncoder(),
                    # 'JamesSteinEncoder': ce.JamesSteinEncoder(),
                    # 'BaseNEncoder': ce.BaseNEncoder(),
                    # 'SumEncoder': ce.SumEncoder()
                    }
    classifiers_list = []
    for key1, encoder1 in encoder_dict.items():
        for key2, encoder2 in encoder_dict.items():
            classifier2 = get_classifier(key2, key1, encoder2, encoder1, clf_df, target_label)
            classifiers_list.append([key1, key2, classifier2])
        classifier1 = get_classifier(key1, key2, encoder1, encoder2, clf_df, target_label)
        classifiers_list.append([key1, key2, classifier1])

    return classifiers_list


def run_model_representation(df, clf_df):
    target_label = "safety"
    classifiers_list = preprocessing_pipeline(clf_df, target_label)
    encoders_comparison_df = pd.DataFrame(columns=['NominalEncoder', 'OrdinalEncoder', 'FinalScore'])
    i = 0
    for element in classifiers_list:
        key1 = element[0]
        key2 = element[1]
        classifier = element[2]
        X = df.drop(target_label, axis=1)
        y = df[target_label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)

        encoders_comparison_df.at[i, 'NominalEncoder'] = key1
        encoders_comparison_df.at[i, 'OrdinalEncoder'] = key2
        encoders_comparison_df.at[i, 'FinalScore'] = score
        i += 1

    encoders_comparison_df.to_csv('encoders_comparison.csv', sep=',', header=True)
    # print(encoders_comparison_df)

