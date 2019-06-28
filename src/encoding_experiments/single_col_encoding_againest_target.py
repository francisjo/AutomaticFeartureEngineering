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


def get_multi_classifier(key, encoder, single_col, numeric_list):
    numeric_transformer = Pipeline(
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
            (key, encoder)
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=
        [
            ('single_col', single_col_transformer, [single_col]),
            ('num', numeric_transformer, numeric_list)
        ]
    )

    return preprocessor


def apply_multiple_encoders_for_one_column_against_others(single_col, numeric_list):

    encoder_dict = main_dicts.get_encoder_dict()
    classifiers_list = []
    for key, encoder in encoder_dict.items():
        classifier = get_multi_classifier(key, encoder, single_col, numeric_list)
        classifiers_list.append([key, classifier])
    return classifiers_list


def change_column_type(df, cat_list):
    for col in df.columns:
        if col in cat_list:
            df[col].astype('object', copy=False)
    return df


def best_encoders_dict(dataframe):
    df1 = dataframe.loc[dataframe.groupby(['ColumnName'], as_index=False)['Score'].idxmax()]
    best_encoders_dict = df1.groupby('Encoder')['ColumnName'].apply(lambda g: g.values.tolist()).to_dict()
    return best_encoders_dict


def best_encoders_dict_dfs(ds_names, encoders_comparison_df):
    best_encoders_dict_dfs = {}
    for ds_name in ds_names:
        df = encoders_comparison_df[encoders_comparison_df['DataSetName'] == ds_name]
        encoders_dict = best_encoders_dict(df)
        best_encoders_dict_dfs[ds_name] = encoders_dict
    return best_encoders_dict_dfs


def run_best_encoding_methods(groundtruth_dict, datasets_dict, encoders_comparison_df):
    ds_names = encoders_comparison_df['DataSetName'].unique()
    encoders_dict_dfs = best_encoders_dict_dfs(ds_names, encoders_comparison_df)
    target_dict = main_dicts.get_target_variables_dicts()
    for ds_key, df in datasets_dict.items():
        target = target_dict[ds_key]
        ground_truth = groundtruth_dict[ds_key]
        numeric_list = [x for x in ground_truth if ground_truth[x] == 'Numerical']
        categorical_list = [x for x in ground_truth if ground_truth[x] != 'Numerical']
        if target in numeric_list:
            numeric_list.remove(target)
        df = change_column_type(df, categorical_list)
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
        best_encoders_dict = encoders_dict_dfs[ds_key]
        numeric_transformer = Pipeline(
            steps=
            [
                ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean', copy=False)),
                ('scaler', StandardScaler())
            ]
        )

        encoder_dict = main_dicts.get_encoder_dict()
        preprocessor = ColumnTransformer(
            transformers=
            [
                ('num', numeric_transformer, numeric_list),
            ]
        )
        for key, listcolumns in best_encoders_dict.items():
            encoder = encoder_dict[key]
            transformer = Pipeline(
                steps=
                [
                    ('imputer', SimpleImputer(strategy="most_frequent", copy=False)),
                    (key, encoder)
                ]
            )
            preprocessor.transformers.append((key, transformer, listcolumns))
        classifier = DecisionTreeClassifier(random_state=23)
        X = df.drop(target, axis=1)
        y = df[target]
        X = preprocessor.fit_transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print(ds_key, score)


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


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
        df = change_column_type(df, categorical_list)
        for item in categorical_list:
            if item != target:
                if item in df.columns:
                    col_type = ground_truth[item]
                    single_col = item
                    nuuniquevalues = df[single_col].nunique()
                    preprocessor_list = apply_multiple_encoders_for_one_column_against_others(single_col, numeric_list)
                    for element in preprocessor_list:
                        key = element[0]
                        preprocessor = element[1]
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
                        encoders_comparison_df.at[i, 'Encoder'] = key
                        encoders_comparison_df.at[i, 'Cardinality'] = nuuniquevalues
                        encoders_comparison_df.at[i, 'Score'] = score
                        encoders_comparison_df.at[i, 'Roc_auc_score'] = roc_auc_score
                        i += 1
        #run_best_encoding_methods(groundtruth_dict, datasets_dict, encoders_comparison_df)
        #file_name ="multiple_encoders_for_all_"+ ds_key + ".csv"
        #encoders_comparison_df.to_csv(file_name, sep=',', header=True)
    encoders_comparison_df.to_csv('single_col_against_target_2806.csv', sep=',', header=True)


#multiple_encoders_for_all_columns()

'''
datasets_dict = ld.load_data_online()
groundtruth_dict = main_dicts.get_groundtruth_dict()
encoders_comparison_df = pd.read_csv('/home/basha/PycharmProjects/DSA_Project/AutomaticFeartureEngineering/src/encoding_experiments/results/single_col_against_target_2806.csv')
run_best_encoding_methods(groundtruth_dict, datasets_dict, encoders_comparison_df)
'''
