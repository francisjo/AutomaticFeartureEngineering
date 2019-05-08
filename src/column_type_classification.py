import pandas.api.types as ptypes
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from decimal import Decimal

import data_cleaning as cd


def columns_correlation_classification_max(df):
    result = {}
    corr = df.corr()
    for col in df.columns:
        corr_values = corr[col]
        corr_values = corr_values.drop(col)
        max_val = corr_values.max()

        result[col] = round(Decimal(max_val), 3)
    return result


def columns_correlation_classification_min(df):
    result = {}
    corr = df.corr()
    for col in df.columns:
        corr_values = corr[col]
        corr_values = corr_values.drop(col)
        min_val = corr_values.min()

        result[col] = round(Decimal(min_val), 3)
    return result


# Detect categorical and numerical variables by Frequency of unique value in Columns Wih Threshold #
def columns_frequency_classification(df):
    result = {}
    for col in df.columns:
        col_count = len(df[col].unique())
        percent = col_count / df[col].count()
        result[col] = round(Decimal(percent), 3)
    return result


# Detect categorical and numerical variables by Frequency of unique value in Columns Wih Threshold #
def columns_distribution_classification(df):
    result = {}
    for col in df.columns:
        value_count = df.groupby(col)[col].count()
        col_dist = value_count / df[col].count()
        max_val = col_dist.max()
        result[col] = round(Decimal(max_val), 3)
    return result


def encode_string_column(df):
    dataframe = pd.DataFrame()
    for col in df.columns:
        if ptypes.is_string_dtype(df[col]):
            cd.fill_null_data(df, col, 'string')
            le = LabelEncoder()
            dataframe[col] = le.fit_transform(df[col])
        if ptypes.is_numeric_dtype(df[col]):
            cd.fill_null_data(df, col, 'numeric')
            dataframe[col] = df[col]
    return dataframe


def get_column_type(df):
    result = {}
    for col in df.columns:
        result[col] = df[col].dtype
    return result


def get_numeric_nominal_ordinal_cols(df_dict):
    summarized_dfs = get_summarized_df(df_dict)
    summarized_df, numeric_cols, nominal_cols, ordinal_cols = final_cols_classification(summarized_df)
    return summarized_dfs, numeric_cols, nominal_cols, ordinal_cols

def get_dataset_groundtruth():
    dict_car = {"make": "nominal",
            "fuel_type": "nominal",
            "aspiration": "nominal",
            "num_of_doors": "discrete",
            "body_style": "nominal",
            "drive_wheels": "nominal",
            "engine_location": "binary",
            "wheel_base": "continuous",
            "length": "continuous",
            "width": "continuous",
            "height": "continuous",
            "height": "continuous",
            "engine_type": "nominal",
            "num_of_cylinders": "discrete",
            "engine_size": "continuous",
            "fuel_system": "nominal",
            "compression_ratio": "continuous",
            "horsepower": "continuous",
            "peak_rpm": "continuous",
            "city_mpg": "continuous",
            "highway_mpg": "continuous",
            "price": "continuous"
            }
    dict_titanic = {"PassengerId": "discrete",
                "Survived": "binary",
                "Pclass": "ordinal",
                "Name": "nominal",
                "Sex": "binary",
                "Age": "discrete",
                "SibSp": "continuous",
                "Parch": "discrete",
                "Ticket": "nominal",
                "Fare": "continuous",
                "Cabin": "ordinal",
                "Embarked": "nominal",
                }
    dict_heart = {"age":"discrete",
                  "sex": "binary",
                  "cp": "nominal",
                  "trestbps": "continuous",
                  "chol": "continuous",
                  "fbs": "binary",
                  "restecg": "nominal",
                  "thalach": "continuous",
                  "exang": "binary",
                  "oldpeak": "continuous",
                  "slope": "ordinal",
                  "ca": "nominal",
                  "thal": "ordinal",
                  "target": "binary",
                  }
    return dict_car


def get_summarized_df(df_dict):
    dict_car = {"make": "nominal",
                "fuel_type": "nominal",
                "aspiration": "nominal",
                "num_of_doors": "discrete",
                "body_style": "nominal",
                "drive_wheels": "nominal",
                "engine_location": "binary",
                "wheel_base": "continuous",
                "length": "continuous",
                "width": "continuous",
                "height": "continuous",
                "height": "continuous",
                "engine_type": "nominal",
                "num_of_cylinders": "discrete",
                "engine_size": "continuous",
                "fuel_system": "nominal",
                "compression_ratio": "continuous",
                "horsepower": "continuous",
                "peak_rpm": "continuous",
                "city_mpg": "continuous",
                "highway_mpg": "continuous",
                "price": "continuous"
                }
    dict_titanic = {"PassengerId": "discrete",
                    "Survived": "binary",
                    "Pclass": "ordinal",
                    "Name": "nominal",
                    "Sex": "binary",
                    "Age": "discrete",
                    "SibSp": "continuous",
                    "Parch": "discrete",
                    "Ticket": "nominal",
                    "Fare": "continuous",
                    "Cabin": "ordinal",
                    "Embarked": "nominal",
                    }
    dict_heart = {"age": "discrete",
                  "sex": "binary",
                  "cp": "nominal",
                  "trestbps": "continuous",
                  "chol": "continuous",
                  "fbs": "binary",
                  "restecg": "nominal",
                  "thalach": "continuous",
                  "exang": "binary",
                  "oldpeak": "continuous",
                  "slope": "ordinal",
                  "ca": "nominal",
                  "thal": "ordinal",
                  "target": "binary",
                  }
    summarized_dfs = pd.DataFrame()
    for item, value in df_dict.items():
        cols_type_dict = get_column_type(value)
        encoded_df = encode_string_column(value)
        cols_dist_dict = columns_distribution_classification(encoded_df)
        cols_freq_dict = columns_frequency_classification(encoded_df)
        cols_corr_dict_max = columns_correlation_classification_max(encoded_df)
        cols_corr_dict_min = columns_correlation_classification_min(encoded_df)
        df_dicts = [cols_dist_dict, cols_freq_dict, cols_corr_dict_min, cols_corr_dict_max, cols_type_dict, {}]
        summarized_df = pd.DataFrame(df_dicts)
        summarized_df["Method"] = ["Dist", "Freq", "Corr_Min", "Corr_Max", "D-Type", "Cls-Result"]
        summarized_df = summarized_df.set_index("Method")
        summarized_df_T = summarized_df.T.reset_index()
        if item == "titanic":
            for col_name in summarized_df_T['index']:
                result = dict_titanic.get(col_name)
                summarized_df_T[summarized_df_T["index"] == col_name, "Cls-Result"] = result
        elif item == "heart":
            for col_name in summarized_df_T['index']:
                result = dict_heart.get(col_name)
                summarized_df_T[summarized_df_T["index"] == col_name, "Cls-Result"] = result
        elif item == "car":
            for col_name in summarized_df_T['index']:
                result = dict_car.get(col_name)
                summarized_df_T[summarized_df_T["index"] == col_name, "Cls-Result"] = result
        summarized_dfs = pd.concat([summarized_dfs,summarized_df_T])

    return summarized_dfs


def get_freq(summarized_df, col, threshold_freq):
    if summarized_df.get_value('Freq', col) >= threshold_freq:
        freq = True    # High
    else:
        freq = False    # Low
    return freq


def get_dist(summarized_df, col, threshold_dist):
    if summarized_df.get_value('Dist', col) >= threshold_dist:
        dist = True     # High
    else:
        dist = False    # Low
    return dist


def get_corr(summarized_df, col, corr_min, corr_max):
    if summarized_df.get_value('Corr_Min', col) < corr_min or summarized_df.get_value('Corr_Max', col) > corr_max:
        corr = True     # High (Positive [closer to one] or Negative [closer to negative one])
    else:
        corr = False    # Low (Closer to Zero)
    return corr


def final_cols_classification(summarized_df):
    numeric_cols = []
    nominal_cols = []
    ordinal_cols = []
    threshold_dist = 0.3
    threshold_freq = 0.2
    corr_min = -0.5
    corr_max = 0.5
    for col in summarized_df.columns:
        freq = get_freq(summarized_df, col, threshold_freq)
        dist = get_dist(summarized_df, col, threshold_dist)
        corr = get_corr(summarized_df, col, corr_min, corr_max)
        is_str = ptypes.is_string_dtype(summarized_df.get_value('D-Type', col))

        if not freq and dist and corr:                          # low freq & high dist & high corr [Str&Num] ==> Ordinal
            ordinal_cols.append(col)
            summarized_df.set_value('Cls-Result', col, 'Ordinal')
        elif freq and not dist and not corr and is_str:         # high freq & low dist & low corr & string   ==> Nominal
            nominal_cols.append(col)
            summarized_df.set_value('Cls-Result', col, 'Nominal')
        elif not freq and dist and not corr and is_str:         # low freq & high dist & low corr & string   ==> Nominal
            nominal_cols.append(col)
            summarized_df.set_value('Cls-Result', col, 'Nominal')
        elif freq and not dist and corr and is_str:             # high freq & low dist & high corr & string  ==> Nominal
            nominal_cols.append(col)
            summarized_df.set_value('Cls-Result', col, 'Nominal')
        elif not freq and not dist and not corr and is_str:     # low freq & low dist & low corr & string  ==> Nominal
            nominal_cols.append(col)
            summarized_df.set_value('Cls-Result', col, 'Nominal')
        elif not freq and dist and not corr and not is_str:     # low freq & high dist & low corr & number   ==> Nominal
            nominal_cols.append(col)
            summarized_df.set_value('Cls-Result', col, 'Nominal')
        elif freq and not dist and corr and not is_str:         # high freq & low dist & high corr & number  ==> Numeric
            numeric_cols.append(col)
            summarized_df.set_value('Cls-Result', col, 'Numeric')
        elif freq and not dist and not corr and not is_str:     # high freq & low dist & low corr & number   ==> Numeric
            numeric_cols.append(col)
            summarized_df.set_value('Cls-Result', col, 'Numeric')
        elif not freq and not dist and not corr and not is_str:  # low freq & low dist & low corr & numeric  ==> Numeric
            numeric_cols.append(col)
            summarized_df.set_value('Cls-Result', col, 'Numeric')

    return summarized_df, numeric_cols, nominal_cols, ordinal_cols


