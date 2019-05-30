from decimal import Decimal
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import pandas.api.types as ptypes
import numpy as np
import word2vec_load as w2v


# Compute the maximum correlation of values inside each column in the provided data-frame #
def columns_correlation_classification_max(df):
    result = {}
    corr = df.corr()
    for col in df.columns:
        corr_values = corr[col]
        corr_values = corr_values.drop(col)
        max_val = corr_values.max()
        result[col] = round(Decimal(max_val), 3)
    return result


# Compute the minimum correlation of values inside each column in the provided data-frame #
def columns_correlation_classification_min(df):
    result = {}
    corr = df.corr()
    for col in df.columns:
        corr_values = corr[col]
        corr_values = corr_values.drop(col)
        min_val = corr_values.min()

        result[col] = round(Decimal(min_val), 3)
    return result


# Compute the strong correlation of values inside each column in the provided data-frame #
def columns_correlation_classification_strong(df):
    result = {}
    corr = df.corr()
    for col in df.columns:
        corr_values = corr[col]
        corr_values = corr_values.drop(col)
        max_val = corr_values.max()
        min_val = corr_values.min()
        max_val_abs = [max_val, abs(max_val)]
        min_val_abs = [min_val, abs(min_val)]
        strong_value = max(max_val_abs[1], min_val_abs[1])
        if strong_value == max_val_abs[1]:
            strong_value = max_val_abs[0]
        else:
            strong_value = min_val_abs[0]
        result[col] = round(Decimal(strong_value), 3)
    return result



# Compute the correlation of values inside each column in the provided data-frame by SpearManR method #
def columns_correlation_spearman_r(df):
    result = {}
    for col1 in df.columns:
        spearmanr_values = pd.Series()
        for col2 in df.columns:
            if col1 != col2:
                value = stats.spearmanr(df[col1], df[col2])[0]
                spearmanr_values[col2] = value
        max_val = spearmanr_values.max()
        min_val = spearmanr_values.min()
        max_val_abs = [max_val, abs(max_val)]
        min_val_abs = [min_val, abs(min_val)]
        strong_value = max(max_val_abs[1], min_val_abs[1])
        if strong_value == max_val_abs[1]:
            strong_value = max_val_abs[0]
        else:
            strong_value = min_val_abs[0]
        result[col1] = round(Decimal(strong_value), 3)
    return result


def columns_correlation_spearman_r_test(df):
    result = {}
    resultwithname = {}
    for col1 in df.columns:
        spearmanr_values = pd.Series()
        for col2 in df.columns:
            if col1 != col2:
                value = stats.spearmanr(df[col1], df[col2])[0]
                spearmanr_values[col2] = [value, col2]
        max_val = spearmanr_values.max()
        min_val = spearmanr_values.min()
        max_val_abs = [max_val[0], abs(max_val[0]), max_val[1]]
        min_val_abs = [min_val[0], abs(min_val[0]), min_val[1]]
        strong_value = max(max_val_abs[1], min_val_abs[1])
        if strong_value == max_val_abs[1]:
            strong_value = max_val_abs[0]
            strong_name = max_val_abs[2]
        else:
            strong_value = min_val_abs[0]
            strong_name = min_val_abs[2]
        result[col1] = round(Decimal(strong_value), 3)
        resultwithname[col1] = strong_name
    return result, resultwithname


def columns_correlation_chi2(df, cols_freq_dict):
    result_isdependent = {}
    result_critical = {}
    result_stat = {}
    result_dof = {}
    for col1 in df.columns:
        if cols_freq_dict[col1] < 0.20:
            chi2_values = pd.DataFrame()
            for col2 in df.columns:
                if col1 != col2:
                    #if(df[col1].nunique() - 1) * (df[col1].nunique() - 1) <= 100:
                    if cols_freq_dict[col2] < 0.20:
                        dfcrosstab = pd.crosstab(df[col1], df[col2])
                        stat, p, dof, expected = chi2_contingency(dfcrosstab)
                        prob = 0.95
                        critical = chi2.ppf(prob, dof)
                        if abs(stat) >= critical:
                            chi2_values[col2] = [1, critical, stat, dof]
                        else:
                            chi2_values[col2] = [0, critical, stat, dof]
            chi2_values_T = chi2_values.T
            if len(chi2_values_T) != 0:
                dependent_min = chi2_values_T[chi2_values_T[0] == 1].groupby(chi2_values_T[0]).min()
                if len(dependent_min) != 0:
                    result_isdependent[col1] = dependent_min.iloc[0, 0]
                    result_critical[col1] = round(dependent_min.iloc[0, 1], 3)
                    result_stat[col1] = round(dependent_min.iloc[0, 2], 3)
                    result_dof[col1] = dependent_min.iloc[0, 3]
                else:
                    independent_min = chi2_values_T[chi2_values_T[0] == 0].groupby(chi2_values_T[0]).min()
                    if len(independent_min) != 0:
                        result_isdependent[col1] = independent_min.iloc[0, 0]
                        result_critical[col1] = round(Decimal(independent_min.iloc[0, 1]), 3)
                        result_stat[col1] = round(Decimal(independent_min.iloc[0, 2]), 3)
                        result_dof[col1] = independent_min.iloc[0, 3]
            else:
                result_isdependent[col1] = 0
                result_critical[col1] = 0
                result_stat[col1] = 0
                result_dof[col1] = 0

    return result_isdependent, result_critical, result_stat, result_dof


# Compute the frequency of values inside each column in the provided data-frame #
def columns_frequency_classification(df):
    result = {}
    for col in df.columns:
        col_count = len(df[col].unique())
        percent = col_count / df[col].count()
        result[col] = round(Decimal(percent), 3)
    return result


# Compute the distribution of values inside each column in the provided data-frame #
def columns_distribution_classification(df):
    result = {}
    for col in df.columns:
        value_count = df.groupby(col)[col].count()
        col_dist = value_count / df[col].count()
        max_val = col_dist.max()
        result[col] = round(Decimal(max_val), 3)
    return result


def word2vec_distances(df):
    model = w2v.model
    result_mean = {}
    result_std = {}
    for col in df.columns:
        if ptypes.is_string_dtype(df[col]):
            my_words = df[col].unique()
            my_words = [word for word in my_words if word in model]  # filter out words not in model
            if len(my_words) < 2:
                result_mean[col] = 0.0
                result_std[col] = 0.0
                continue
            # Make a list of all word-to-word distances [each as a tuple of (word1,word2,dist)]
            dists = []
            # Method 1 to find distances: use gensim to get the similarity between each word pair
            for i1, word1 in enumerate(my_words):
                for i2, word2 in enumerate(my_words):
                    if i1 >= i2:
                        continue
                    cosine_distance = model.distance(word1, word2)
                    dists.append(cosine_distance)
            result_mean[col] = round(Decimal(np.mean(dists)), 3)
            result_std[col] = round(Decimal(np.std(dists)), 3)
        elif ptypes.is_numeric_dtype(df[col]):
            result_mean[col] = 0.0
            result_std[col] = 0.0
    return result_mean, result_std,

