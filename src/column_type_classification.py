import pandas.api.types as ptypes
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

import data_cleaning as cd
import statistical_functions as sf


def get_summarized_df(df_dict):
    summarized_dfs = pd.DataFrame()
    for item, value in df_dict.items():
        cols_type_dict = get_column_type(value)
        encoded_df = encode_string_column(value)
        cols_dist_dict = sf.columns_distribution_classification(encoded_df)
        cols_freq_dict = sf.columns_frequency_classification(encoded_df)
        cols_corr_dict_max = sf.columns_correlation_classification_max(encoded_df)
        cols_corr_dict_min = sf.columns_correlation_classification_min(encoded_df)
        cols_corr_dict_strong = sf.columns_correlation_classification_strong(encoded_df)
        cols_spearman_dict = sf.columns_correlation_spearman_r(encoded_df)
        cols_isbinary = column_is_binary(encoded_df)
        #cols_is_dependent,cols_critical,cols_stat,cols_dof = sf.columns_correlation_chi2(encoded_df,cols_freq_dict)
        df_dicts = [cols_dist_dict, cols_freq_dict, cols_corr_dict_min, cols_corr_dict_max, cols_corr_dict_strong, cols_spearman_dict, cols_isbinary, cols_type_dict, {}]
        summarized_df = pd.DataFrame(df_dicts)
        summarized_df["Method"] = ["Dist", "Freq", "Corr_Min", "Corr_Max", "Corr_Strong","Corr_Spearman", "Is_binary", "D-Type", "Cls-Result"]
        summarized_df = summarized_df.set_index("Method")
        summarized_df_T = summarized_df.T.reset_index()
        get_ground_truth(summarized_df_T, item)
        summarized_dfs = pd.concat([summarized_dfs, summarized_df_T])

    cls_result_replace_map = {"Binary": 0, "Numerical": 1, "Nominal": 2, "Ordinal": 3}
    summarized_dfs["Cls-Result"].replace(cls_result_replace_map, inplace=True)
    dtype_replace_map = {"bool": 0, "object": 1, "int64": 2, "float64": 3, "datetime64": 4}
    summarized_dfs["D-Type"].replace(dtype_replace_map, inplace=True)
    one_hot = pd.get_dummies(summarized_dfs['D-Type'])
    summarized_dfs = summarized_dfs.drop('D-Type', axis=1)
    #summarized_dfs = summarized_dfs.join(one_hot)
    summarized_dfs = pd.concat([summarized_dfs, one_hot],axis=1)
    summarized_dfs = summarized_dfs.drop(["Corr_Min", "Corr_Max","Corr_Strong"], axis=1)
    return summarized_dfs


def get_ground_truth(summarized_df_T, item):
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
                  "sex": "Binary",
                  "capital-gain": "Numerical",
                  "capital-loss": "Numerical",
                  "hours-per-week": "Numerical",
                  "native-country": "Nominal",
                  "class": "Binary",
             },
         "car":
             {
                "make": "Nominal",
                "fuel_type": "Binary",
                "aspiration": "Binary",
                "num_of_doors": "Numerical",
                "body_style": "Nominal",
                "drive_wheels": "Binary",
                "engine_location": "Binary",
                "wheel_base": "Numerical",
                "length": "Numerical",
                "width": "Numerical",
                "height": "Numerical",
                "engine_type": "Nominal",
                "num_of_cylinders": "Ordinal",
                "engine_size": "Numerical",
                "fuel_system": "Nominal",
                "compression_ratio": "Numerical",
                "horsepower": "Numerical",
                "peak_rpm": "Numerical",
                "city_mpg": "Numerical",
                "highway_mpg": "Numerical",
                "price": "Numerical",
                "curb_weight": "Numerical",
                "num_of_doors_num": "Binary",
                "num_of_cylinders_num": "Nominal"
            },
         "titanic":
             {
                "PassengerId": "Numerical",
                "Survived": "Binary",
                "Pclass": "Ordinal",
                "Name": "Nominal",
                "Sex": "Binary",
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
                "LOCATION": "Nominal",
                "ERECTED": "Numerical",
                "PURPOSE": "Nominal",
                "LENGTH": "Numerical",
                "LANES": "Numerical",
                "CLEAR-G": "Binary",
                "T-OR-D": "Binary",
                "MATERIAL": "Nominal",
                "SPAN": "Ordinal",
                "REL-L": "Nominal",
                "binaryClass": "Binary"
             },
         "heart":
             {
                "age": "Numerical",
                "sex": "Binary",
                "cp": "Nominal",
                "trestbps": "Numerical",
                "chol": "Numerical",
                "fbs": "Binary",
                "restecg": "Nominal",
                "thalach": "Numerical",
                "exang": "Binary",
                "oldpeak": "Numerical",
                "slope": "Ordinal",
                "ca": "Nominal",
                "thal": "Ordinal",
                "target": "Binary",
             }
    }
    for col_name in summarized_df_T['index']:
        result = groundtruth_dict[item].get(col_name)
        summarized_df_T.loc[summarized_df_T["index"] == col_name, "Cls-Result"] = result


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