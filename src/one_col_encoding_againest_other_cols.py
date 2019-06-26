from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import load_datasets as ld
import numpy as np
import pandas as pd
import main_dicts


def get_multi_classifier(key1, key2, encoder1, encoder2, single_col, other_cols, numeric_list):
    numeric_transformer = Pipeline  (
        steps=
        [
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean', copy=False)),
            ('scaler', StandardScaler())
        ]
    )

    single_col_transformer = Pipeline(
        steps=
        [
            ('imputer', SimpleImputer(missing_values=np.nan, strategy="most_frequent", copy=False)),
            (key1, encoder1)
        ]
    )

    other_cols_transformer = Pipeline(
        steps=
        [
            ('imputer', SimpleImputer(missing_values=np.nan, strategy="most_frequent", copy=False)),
            (key2, encoder2)
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=
        [
            ('single_col', single_col_transformer, [single_col]),
            ('num', numeric_transformer, numeric_list),
            ('other_cols', other_cols_transformer, other_cols)
        ]
    )

    return preprocessor


def apply_multiple_encoders_for_one_column_against_others(single_col, other_cols, numeric_list):

    encoder_dict = main_dicts.get_encoder_dict()
    classifiers_list = []
    for key1, encoder1 in encoder_dict.items():
        for key2, encoder2 in encoder_dict.items():
            classifier2 = get_multi_classifier(key1, key2, encoder1, encoder2, single_col, other_cols, numeric_list)
            classifiers_list.append([key1, key2, classifier2])
    return classifiers_list


def multiple_encoders_for_all_columns():
    datasets_dict = ld.load_data_online()
    groundtruth_dict = main_dicts.get_groundtruth_dict()
    target_dict = main_dicts.get_target_variables_dicts()
    encoders_comparison_df = pd.DataFrame(columns=['DataSetName', 'ColumnName', 'ColumnType', 'Encoder', 'EncoderForOthers', 'Cardinality', 'Score'])
    i = 0
    for ds_key, df in datasets_dict.items():
        ground_truth = groundtruth_dict[ds_key]
        target = target_dict[ds_key]
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
        numeric_list = [x for x in ground_truth if ground_truth[x] == 'Numerical']
        categorical_list = [x for x in ground_truth if ground_truth[x] != 'Numerical']
        if target in categorical_list:
            categorical_list.remove(target)
        if target in numeric_list:
            numeric_list.remove(target)
        for item in categorical_list:
            if item != target:
                if item in df.columns:
                    col_type = ground_truth[item]
                    single_col = item
                    nuuniquevalues = df[single_col].nunique()
                    other_cols = categorical_list.copy()
                    other_cols.remove(single_col)
                    classifiers_list = apply_multiple_encoders_for_one_column_against_others(single_col, other_cols, numeric_list)
                    for element in classifiers_list:
                        key1 = element[0]
                        key2 = element[1]
                        preprocessor = element[2]
                        classifier = DecisionTreeClassifier(random_state=23)
                        X = df.drop(target, axis=1)
                        y = df[target]
                        X = preprocessor.fit_transform(X, y)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
                        classifier.fit(X_train, y_train)
                        # score = classifier.score(X_test, y_test)
                        score = roc_auc_score(y, classifier.predict(X))
                        encoders_comparison_df.at[i, 'DataSetName'] = ds_key
                        encoders_comparison_df.at[i, 'ColumnName'] = single_col
                        encoders_comparison_df.at[i, 'ColumnType'] = col_type
                        encoders_comparison_df.at[i, 'Encoder'] = key1
                        encoders_comparison_df.at[i, 'EncoderForOthers'] = key2
                        encoders_comparison_df.at[i, 'Cardinality'] = nuuniquevalues
                        encoders_comparison_df.at[i, 'Score'] = score
                        i += 1
        file_name ="multiple_encoders_for_all_"+ ds_key + ".csv"
        encoders_comparison_df.to_csv(file_name, sep=',', header=True)


    #encoders_comparison_df.to_csv('multiple_encoders_for_all_columns120619.csv', sep=',', header=True)


