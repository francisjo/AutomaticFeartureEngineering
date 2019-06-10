from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas.api.types as ptypes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import category_encoders as ce
import load_datasets as ld
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
    encoder_dict = {'OneHotEncoder': ce.OneHotEncoder(),
                    'BinaryEncoder': ce.BinaryEncoder(),
                    'HashingEncoder': ce.HashingEncoder(),
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


# =========== single encoding againest target variable =============


def single_pipeline(col, col_type):
    encoder_dict = {'OneHotEncoder': ce.OneHotEncoder(),
                    'BinaryEncoder': ce.BinaryEncoder(),
                    'HashingEncoder': ce.HashingEncoder(),
                    # 'OrdinalEncoder': ce.OrdinalEncoder(),
                    # 'PolynomialEncoder': ce.PolynomialEncoder(),
                    # 'TargetEncoder': ce.TargetEncoder(),
                    # 'HelmertEncoder': ce.HelmertEncoder(),
                    # 'JamesSteinEncoder': ce.JamesSteinEncoder(),
                    # 'BaseNEncoder': ce.BaseNEncoder(),
                    # 'SumEncoder': ce.SumEncoder()
                    }
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
                ('classifier', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100000))
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
                    ('classifier', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100000))
                ]
            )
            classifiers_list.append([classifier, key])
    return classifiers_list


def single_encoder_against_target():
    datasets_dict = ld.load_data_online()
    groundtruth_dict = {
        "adult":
            {
                "age": "Numerical",
                "workclass": "Nominal",
                "fnlwgt": "Numerical",
                "education": "Ordinal",
                "education-num": "Ordinal",
                "marital-status": "Nominal",
                "occupation": "Nominal",
                "relationship": "Nominal",
                "race": "Nominal",
                "sex": "Nominal",
                "capital-gain": "Numerical",
                "capital-loss": "Numerical",
                "hours-per-week": "Numerical",
                "native-country": "Nominal",
                "class": "Nominal",
            },
        "car":
            {
                "make": "Nominal",
                "fuel_type": "Nominal",
                "aspiration": "Nominal",
                "body_style": "Nominal",
                "drive_wheels": "Nominal",
                "engine_location": "Nominal",
                "wheel_base": "Numerical",
                "length": "Numerical",
                "width": "Numerical",
                "height": "Numerical",
                "engine_type": "Nominal",
                "num_of_cylinders": "Nominal",
                "engine_size": "Numerical",
                "fuel_system": "Nominal",
                "compression_ratio": "Numerical",
                "horsepower": "Numerical",
                "peak_rpm": "Numerical",
                "city_mpg": "Numerical",
                "highway_mpg": "Numerical",
                "price": "Numerical",
                "curb_weight": "Numerical",
                "num_of_doors_num": "Nominal",
                "num_of_cylinders_num": "Numerical"
            },
        "titanic":
            {
                "PassengerId": "Nominal",
                "Survived": "Nominal",
                "Pclass": "Ordinal",
                "Name": "Nominal",
                "Sex": "Nominal",
                "Age": "Numerical",
                "SibSp": "Numerical",
                "Parch": "Numerical",
                "Ticket": "Nominal",
                "Fare": "Numerical",
                "Cabin": "Ordinal",
                "Embarked": "Nominal",
            },
        "bridges":
            {
                "IDENTIF": "Nominal",
                "RIVER": "Nominal",
                "LOCATION": "Numerical",
                "ERECTED": "Numerical",
                "PURPOSE": "Nominal",
                "LENGTH": "Numerical",
                "LANES": "Numerical",
                "CLEAR-G": "Nominal",
                "T-OR-D": "Nominal",
                "MATERIAL": "Nominal",
                "SPAN": "Ordinal",
                "REL-L": "Nominal",
                "binaryClass": "Nominal"
            },
        "heart":
            {
                "age": "Numerical",
                "sex": "Nominal",
                "cp": "Nominal",
                "trestbps": "Numerical",
                "chol": "Numerical",
                "fbs": "Nominal",
                "restecg": "Nominal",
                "thalach": "Numerical",
                "exang": "Nominal",
                "oldpeak": "Numerical",
                "slope": "Ordinal",
                "ca": "Nominal",
                "thal": "Ordinal",
                "target": "Nominal",
            },
        "audiology":
            {
                "air": "Ordinal",
                "ar_c": "Nominal",
                "ar_u": "Nominal",
                "o_ar_c": "Nominal",
                "o_ar_u": "Nominal",
                "speech": "Ordinal",
                "indentifier": "Nominal",
                "class": "Nominal"
            },
        "car1":
            {
                "buying": "Nominal",
                "maint": "Ordinal",
                "doors": "Numerical",
                "persons": "Numerical",
                "lug_boot": "Ordinal",
                "safety": "Ordinal"
            },
        "random":
            {
                "Color": "Nominal",
                "Size": "Ordinal",
                "Act": "Nominal",
                "Age": "Nominal",
                "Inflated": "Nominal"
            }
    }
    target_dict = {"adult": "class",
                   "car": "price",
                   "titanic": "Survived",
                   "bridges": "binaryClass",
                   "heart": "target",
                   "audiology": "class",
                   "car1": "safety",
                   "random": "Inflated"}
    encoders_comparison_df = pd.DataFrame(columns=['DataSetName', 'ColumnName', 'ColumnType', 'Encoder', 'Score'])

    i = 0
    for ds_key, df in datasets_dict.items():
        ground_truth = groundtruth_dict[ds_key]
        target = target_dict[ds_key]
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
                    encoders_comparison_df.at[i, 'Score'] = score
                    i += 1
    encoders_comparison_df.to_csv('single_encoder_against_target100619.csv', sep=',', header=True)


# ============================ One column against other columns ==========================


def get_multi_classifier(key1, key2, encoder1, encoder2, single_col, other_cols, numeric_list):

    numeric_transformer = Pipeline(
        steps=
        [
            ('imputer', SimpleImputer(missing_values=np.NaN, strategy='mean', copy=False)),
            ('scaler', StandardScaler())
        ]
    )
    single_col_transformer = Pipeline(
        steps=
        [
            ('imputer', SimpleImputer(strategy="most_frequent", copy=False)),
            (key1, encoder1)
        ]
    )
    other_cols_transformer = Pipeline(
        steps=
        [
            ('imputer', SimpleImputer(strategy="most_frequent", copy=False)),
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
    classifier = Pipeline(
        steps=
        [
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100000))
        ]
    )
    return classifier


def apply_multiple_encoders_for_one_column_against_others(single_col, other_cols, numeric_list):
    encoder_dict = {'OneHotEncoder': ce.OneHotEncoder(),
                    'BinaryEncoder': ce.BinaryEncoder(),
                    'HashingEncoder': ce.HashingEncoder(),
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
            classifier2 = get_multi_classifier(key2, key1, encoder2, encoder1, single_col, other_cols, numeric_list)
            classifiers_list.append([key1, key2, classifier2])
        classifier1 = get_multi_classifier(key1, key2, encoder1, encoder2, single_col, other_cols, numeric_list)
        classifiers_list.append([key1, key2, classifier1])
    return classifiers_list


def multiple_encoders_for_all_columns():
    datasets_dict = ld.load_data_online()
    groundtruth_dict = {
        "adult":
            {
                "age": "Numerical",
                "workclass": "Nominal",
                "fnlwgt": "Numerical",
                "education": "Ordinal",
                "education-num": "Ordinal",
                "marital-status": "Nominal",
                "occupation": "Nominal",
                "relationship": "Nominal",
                "race": "Nominal",
                "sex": "Nominal",
                "capital-gain": "Numerical",
                "capital-loss": "Numerical",
                "hours-per-week": "Numerical",
                "native-country": "Nominal",
                "class": "Nominal",
            },
        "car":
            {
                "make": "Nominal",
                "fuel_type": "Nominal",
                "aspiration": "Nominal",
                "body_style": "Nominal",
                "drive_wheels": "Nominal",
                "engine_location": "Nominal",
                "wheel_base": "Numerical",
                "length": "Numerical",
                "width": "Numerical",
                "height": "Numerical",
                "engine_type": "Nominal",
                "num_of_cylinders": "Nominal",
                "engine_size": "Numerical",
                "fuel_system": "Nominal",
                "compression_ratio": "Numerical",
                "horsepower": "Numerical",
                "peak_rpm": "Numerical",
                "city_mpg": "Numerical",
                "highway_mpg": "Numerical",
                "price": "Numerical",
                "curb_weight": "Numerical",
                "num_of_doors_num": "Nominal",
                "num_of_cylinders_num": "Numerical"
            },
        "titanic":
            {
                "PassengerId": "Nominal",
                "Survived": "Nominal",
                "Pclass": "Ordinal",
                "Name": "Nominal",
                "Sex": "Nominal",
                "Age": "Numerical",
                "SibSp": "Numerical",
                "Parch": "Numerical",
                "Ticket": "Nominal",
                "Fare": "Numerical",
                "Cabin": "Ordinal",
                "Embarked": "Nominal",
            },
        "bridges":
            {
                "IDENTIF": "Nominal",
                "RIVER": "Nominal",
                "LOCATION": "Numerical",
                "ERECTED": "Numerical",
                "PURPOSE": "Nominal",
                "LENGTH": "Numerical",
                "LANES": "Numerical",
                "CLEAR-G": "Nominal",
                "T-OR-D": "Nominal",
                "MATERIAL": "Nominal",
                "SPAN": "Ordinal",
                "REL-L": "Nominal",
                "binaryClass": "Nominal"
            },
        "heart":
            {
                "age": "Numerical",
                "sex": "Nominal",
                "cp": "Nominal",
                "trestbps": "Numerical",
                "chol": "Numerical",
                "fbs": "Nominal",
                "restecg": "Nominal",
                "thalach": "Numerical",
                "exang": "Nominal",
                "oldpeak": "Numerical",
                "slope": "Ordinal",
                "ca": "Nominal",
                "thal": "Ordinal",
                "target": "Nominal",
            },
        "audiology":
            {
                "air": "Ordinal",
                "ar_c": "Nominal",
                "ar_u": "Nominal",
                "o_ar_c": "Nominal",
                "o_ar_u": "Nominal",
                "speech": "Ordinal",
                "indentifier": "Nominal",
                "class": "Nominal"
            },
        "car1":
            {
                "buying": "Nominal",
                "maint": "Ordinal",
                "doors": "Numerical",
                "persons": "Numerical",
                "lug_boot": "Ordinal",
                "safety": "Ordinal"
            },
        "random":
            {
                "Color": "Nominal",
                "Size": "Ordinal",
                "Act": "Nominal",
                "Age": "Nominal",
                "Inflated": "Nominal"
            }
    }
    target_dict = {"adult": "class",
                   "car": "price",
                   "titanic": "Survived",
                   "bridges": "binaryClass",
                   "heart": "target",
                   "audiology": "class",
                   "car1": "safety",
                   "random": "Inflated"}
    encoders_comparison_df = pd.DataFrame(columns=['DataSetName', 'ColumnName', 'ColumnType', 'Encoder', 'EncoderForOthers', 'Score'])
    i = 0
    classifiers_list = []
    for ds_key, df in datasets_dict.items():
        ground_truth = groundtruth_dict[ds_key]
        target = target_dict[ds_key]
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
                    other_cols = categorical_list
                    other_cols.remove(single_col)
                    classifiers_list = apply_multiple_encoders_for_one_column_against_others(single_col, other_cols, numeric_list)
                    for element in classifiers_list:
                        key1 = element[0]
                        key2 = element[1]
                        classifier = element[2]
                        X = df.drop(target, axis=1)
                        y = df[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

                        classifier.fit(X_train, y_train)
                        score = classifier.score(X_test, y_test)

                        encoders_comparison_df.at[i, 'DataSetName'] = ds_key
                        encoders_comparison_df.at[i, 'ColumnName'] = single_col
                        encoders_comparison_df.at[i, 'ColumnType'] = col_type
                        encoders_comparison_df.at[i, 'Encoder'] = key1
                        encoders_comparison_df.at[i, 'EncoderForOthers'] = key2
                        encoders_comparison_df.at[i, 'Score'] = score
                        i += 1
        file_name ="multiple_encoders_for_all_"+ ds_key + ".csv"
        encoders_comparison_df.to_csv(file_name, sep=',', header=True)

    #encoders_comparison_df.to_csv('multiple_encoders_for_all_columns100619.csv', sep=',', header=True)


multiple_encoders_for_all_columns()



