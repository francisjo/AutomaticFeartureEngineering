from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import load_datasets as ld
import numpy as np
import pandas as pd
import main_dicts


def change_column_type(df, cat_list):
    for col in df.columns:
        if col in cat_list:
            df[col].astype('object', copy=False)
    return df


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def get_preprocessor(key1, key2, encoder1, encoder2, single_col, other_cols, numeric_list):
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


def get_encoders_for_one_column_against_others(single_col, other_cols, numeric_list):
    encoder_dict = main_dicts.get_encoder_dict()
    preprocessors_list = []
    for key1, encoder1 in encoder_dict.items():
        for key2, encoder2 in encoder_dict.items():
            preprocessor = get_preprocessor(key1, key2, encoder1, encoder2, single_col, other_cols, numeric_list)
            preprocessors_list.append([key1, key2, preprocessor])
    return preprocessors_list


def one_col_encoding_against_other_cols():
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
        df = change_column_type(df, categorical_list)
        for item in categorical_list:
            if item != target:
                if item in df.columns:
                    col_type = ground_truth[item]
                    single_col = item
                    nuuniquevalues = df[single_col].nunique()
                    other_cols = categorical_list.copy()
                    other_cols.remove(single_col)
                    preprocessors_list = get_encoders_for_one_column_against_others(single_col, other_cols, numeric_list)
                    for element in preprocessors_list:
                        key1 = element[0]
                        key2 = element[1]
                        preprocessor = element[2]
                        classifier = DecisionTreeClassifier(random_state=23)
                        X = df.drop(target, axis=1)
                        y = df[target]
                        X = preprocessor.fit_transform(X, y)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
                        classifier.fit(X_train, y_train)
                        score = classifier.score(X_test, y_test)
                        roc_auc_score = multiclass_roc_auc_score(y_test, classifier.predict(X_test))
                        encoders_comparison_df.at[i, 'DataSetName'] = ds_key
                        encoders_comparison_df.at[i, 'ColumnName'] = single_col
                        encoders_comparison_df.at[i, 'ColumnType'] = col_type
                        encoders_comparison_df.at[i, 'Encoder'] = key1
                        encoders_comparison_df.at[i, 'EncoderForOthers'] = key2
                        encoders_comparison_df.at[i, 'Cardinality'] = nuuniquevalues
                        encoders_comparison_df.at[i, 'Score'] = score
                        encoders_comparison_df.at[i, 'Roc_auc_score'] = roc_auc_score
                        i += 1
        # file_name ="multiple_encoders_for_all_"+ ds_key + ".csv"
        # encoders_comparison_df.to_csv(file_name, sep=',', header=True)
    encoders_comparison_df.to_csv('one_col_encoding_against_other_cols.csv', sep=',', header=True)


