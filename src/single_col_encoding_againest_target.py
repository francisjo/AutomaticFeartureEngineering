from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas.api.types as ptypes
import main_dicts
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import category_encoders as ce
import load_datasets as ld
import numpy as np
import pandas as pd


def single_pipeline(col, col_type):
    encoder_dict = main_dicts.get_encoder_dict()
    classifiers_list = []
    if col_type == "Numerical":
        transformer = Pipeline(
            steps=
            [
                ('imputer', SimpleImputer(missing_values=np.NaN, strategy='mean', copy=False)),
                ('scaler', StandardScaler())
            ]
        )
        classifier = Pipeline(
            steps=
            [
                ('transformer', transformer),
                ('classifier', DecisionTreeClassifier(random_state=23))
            ]
        )
        classifiers_list.append([classifier, "Numerical"])
    else:
        for key, encoder in encoder_dict.items():
            transformer = Pipeline(
                steps=
                [
                    ('imputer', SimpleImputer(strategy="most_frequent", copy=False)),
                    (key, encoder)
                ]
            )
            classifier = Pipeline(
                steps=
                [
                    ('transformer', transformer),
                    ('classifier', DecisionTreeClassifier(random_state=23))
                ]
            )
            classifiers_list.append([classifier, key])
    return classifiers_list


def single_encoder_against_target():
    datasets_dict = ld.load_data_online()
    groundtruth_dict = main_dicts.get_groundtruth_dict()
    target_dict = main_dicts.get_target_variables_dicts()
    encoders_comparison_df = pd.DataFrame(columns=['DataSetName', 'ColumnName', 'ColumnType', 'Encoder', 'Cardinality', 'Score'])

    i = 0
    for ds_key, df in datasets_dict.items():
        ground_truth = groundtruth_dict[ds_key]
        target = target_dict[ds_key]
        numeric_list = [x for x in ground_truth if ground_truth[x] == 'Numerical']
        categorical_list = [x for x in ground_truth if ground_truth[x] != 'Numerical']
        for col in df.columns:
            if col != target:
                col_type = ground_truth[col]
                classifiers_list = single_pipeline(col, col_type)
                for element in classifiers_list:
                    classifier = element[0]
                    enc_key = element[1]
                    X = df[col]
                    X = X.to_frame()
                    y = df[target]
                    if ptypes.is_string_dtype(df[target]):
                        le = LabelEncoder()
                        y = le.fit_transform(df[target])
                        y = pd.Series(y)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

                    classifier.fit(X_train, y_train)
                    score = classifier.score(X_test, y_test)

                    encoders_comparison_df.at[i, 'DataSetName'] = ds_key
                    encoders_comparison_df.at[i, 'ColumnName'] = col
                    encoders_comparison_df.at[i, 'ColumnType'] = col_type
                    encoders_comparison_df.at[i, 'Encoder'] = enc_key
                    encoders_comparison_df.at[i, 'Cardinality'] = enc_key
                    encoders_comparison_df.at[i, 'Score'] = score
                    i += 1
    encoders_comparison_df.to_csv('single_encoder_against_target100619.csv', sep=',', header=True)


