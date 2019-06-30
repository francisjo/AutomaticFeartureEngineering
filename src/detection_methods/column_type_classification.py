import pandas.api.types as ptypes
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import data_cleaning as cd
import statistical_functions as sf
import main_dicts


def get_summarized_df(df_dict):
    summarized_dfs = pd.DataFrame()
    for item, value in df_dict.items():
        cols_type_dict = get_column_type(value)
        value = value.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        word2vec_mean, word2vec_std = sf.word2vec_distances(value)
        doc2vec_norm, doc2vec_vector = sf.doc2vec_vector(value)
        doc2vec_df = pd.DataFrame.from_dict(doc2vec_vector,orient="index").reset_index()
        doc2vec_df.fillna(value=0.0, inplace=True)
        encoded_df = encode_string_column(value)
        cols_values_skewness = sf.column_values_skewness(encoded_df)
        cols_dist_dict = sf.columns_distribution_classification(encoded_df)
        cols_freq_dict = sf.columns_frequency_classification(encoded_df)
        cols_corr_dict_max = sf.columns_correlation_classification_max(encoded_df)
        cols_corr_dict_min = sf.columns_correlation_classification_min(encoded_df)
        cols_corr_dict_strong = sf.columns_correlation_classification_strong(encoded_df)
        cols_spearman_dict, col_names = sf.columns_correlation_spearman_r_test(encoded_df)
        cols_isbinary = column_is_binary(encoded_df)
        #cols_is_dependent,cols_critical,cols_stat,cols_dof = sf.columns_correlation_chi2(encoded_df,cols_freq_dict)
        df_dicts = [cols_dist_dict, cols_freq_dict, cols_corr_dict_min, cols_corr_dict_max, cols_corr_dict_strong, cols_spearman_dict, col_names, word2vec_mean, word2vec_std, doc2vec_norm, cols_values_skewness, cols_isbinary, cols_type_dict, {}]
        summarized_df = pd.DataFrame(df_dicts)
        summarized_df["Method"] = ["Dist", "Freq", "Corr_Min", "Corr_Max", "Corr_Strong", "Corr", "corr_col", "word2vec_mean", "word2vec_std", "doc2vec_norm", "values_skewness", "Is_binary", "D-Type", "Cls-Result"]
        summarized_df = summarized_df.set_index("Method")
        summarized_df_T = summarized_df.T.reset_index()
        summarized_df_T['Df_Name']= item
        #summarized_df_T = summarized_df_T.merge(doc2vec_df, on='index')
        groundtruth_dict = main_dicts.get_groundtruth_dict()
        for col_name in summarized_df_T['index']:
            result = groundtruth_dict[item].get(col_name)
            summarized_df_T.loc[summarized_df_T["index"] == col_name, "Cls-Result"] = result
        summarized_dfs = pd.concat([summarized_dfs, summarized_df_T])

    cls_result_replace_map = {"Numerical": 1, "Nominal": 2, "Ordinal": 3}
    summarized_dfs["Cls-Result"].replace(cls_result_replace_map, inplace=True)
    dtype_replace_map = {"bool": 5555, "object": 1111, "int64": 2222, "float64": 3333, "datetime64": 4444}
    summarized_dfs["D-Type"].replace(dtype_replace_map, inplace=True)
    one_hot = pd.get_dummies(summarized_dfs['D-Type'])
    summarized_dfs = summarized_dfs.drop('D-Type', axis=1)
    # summarized_dfs = summarized_dfs.join(one_hot)
    summarized_dfs = pd.concat([summarized_dfs, one_hot], axis=1)
    summarized_dfs = correct_missing_column(summarized_dfs)
    summarized_dfs = summarized_dfs.drop(["Corr_Min", "Corr_Max", "Corr_Strong", "corr_col", "doc2vec_norm", "Corr", "values_skewness"], axis=1)

    return summarized_dfs


def correct_missing_column(df):
    if 5555 not in df.columns:
        df['5555'] = 0
    if 1111 not in df.columns:
        df['1111'] = 0
    if 2222 not in df.columns:
        df['2222'] = 0
    if 3333 not in df.columns:
        df['3333'] = 0
    if 4444 not in df.columns:
        df['4444'] = 0
    return df


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

