import pandas.api.types as ptypes
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import data_cleaning as cd
import statistical_functions as sf


def get_summarized_df(df_dict):
    dict_adult = {"age": "discrete",
                  "workclass": "nominal",
                  "fnlwgt": "continuous",
                  "education": "ordinal",
                  "education-num": "ordinal",
                  "marital-status": "nominal",
                  "occupation": "nominal",
                  "relationship": "nominal",
                  "race": "nominal",
                  "sex": "binary",
                  "capital-gain": "continuous",
                  "capital-loss": "continuous",
                  "hours-per-week": "discrete",
                  "native-country": "nominal",
                  "class": "binary",
                  }
    dict_car = {"make": "nominal",
                "fuel_type": "binary",
                "aspiration": "binary",
                "num_of_doors": "discrete",
                "body_style": "nominal",
                "drive_wheels": "binary",
                "engine_location": "binary",
                "wheel_base": "continuous",
                "length": "continuous",
                "width": "continuous",
                "height": "continuous",
                "engine_type": "nominal",
                "num_of_cylinders": "ordinal",
                "engine_size": "continuous",
                "fuel_system": "nominal",
                "compression_ratio": "continuous",
                "horsepower": "continuous",
                "peak_rpm": "continuous",
                "city_mpg": "continuous",
                "highway_mpg": "continuous",
                "price": "continuous",
                "curb_weight": "continuous"
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
        cols_dist_dict = sf.columns_distribution_classification(encoded_df)
        cols_freq_dict = sf.columns_frequency_classification(encoded_df)
        cols_corr_dict_max = sf.columns_correlation_classification_max(encoded_df)
        cols_corr_dict_min = sf.columns_correlation_classification_min(encoded_df)
        cols_corr_dict_strong = sf.columns_correlation_classification_strong(encoded_df)
        cols_spearman_dict_strong = sf.columns_correlation_spearman_r(encoded_df)
        cols_isbinary = column_is_binary(encoded_df)
        cols_isdependent,cols_critical,cols_stat,cols_dof = sf.columns_correlation_chi2(encoded_df)
        df_dicts = [cols_dist_dict, cols_freq_dict, cols_corr_dict_min, cols_corr_dict_max, cols_corr_dict_strong, cols_spearman_dict_strong,cols_isdependent,cols_critical,cols_stat,cols_dof, cols_isbinary, cols_type_dict, {}]
        summarized_df = pd.DataFrame(df_dicts)
        summarized_df["Method"] = ["Dist", "Freq", "Corr_Min", "Corr_Max", "Corr_Strong", "Corr_Spearman", "chi2_isdependent", "chi2_critical", "chi2_stat", "chi2_dof", "Is_binary", "D-Type", "Cls-Result"]
        summarized_df = summarized_df.set_index("Method")
        summarized_df_T = summarized_df.T.reset_index()
        if item == "titanic":
            for col_name in summarized_df_T['index']:
                result = dict_titanic.get(col_name)
                summarized_df_T.loc[summarized_df_T["index"] == col_name, "Cls-Result"] = result
        elif item == "heart":
            for col_name in summarized_df_T['index']:
                result = dict_heart.get(col_name)
                summarized_df_T.loc[summarized_df_T["index"] == col_name, "Cls-Result"] = result
        elif item == "car":
            for col_name in summarized_df_T['index']:
                result = dict_car.get(col_name)
                summarized_df_T.loc[summarized_df_T["index"] == col_name, "Cls-Result"] = result
        elif item == "adult":
            for col_name in summarized_df_T['index']:
                result = dict_adult.get(col_name)
                summarized_df_T.loc[summarized_df_T["index"] == col_name, "Cls-Result"] = result
        summarized_dfs = pd.concat([summarized_dfs,summarized_df_T])
    le = LabelEncoder()
    summarized_dfs["Cls-Result"] = le.fit_transform(summarized_dfs["Cls-Result"])
    summarized_dfs["D-Type"] = le.fit_transform(summarized_dfs["D-Type"])
    one_hot = pd.get_dummies(summarized_dfs['D-Type'])
    summarized_dfs = summarized_dfs.drop('D-Type', axis=1)
    summarized_dfs = summarized_dfs.join(one_hot)
    # summarized_dfs = summarized_dfs.drop(["Corr_Min", "Corr_Max"], axis=1))
    return summarized_dfs


def column_is_binary(df):
    result = {}
    for col in df.columns:
        col_count = len(df[col].unique())
        if col_count == 2:
            result[col] = 1
        else:
            result[col] = 0
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