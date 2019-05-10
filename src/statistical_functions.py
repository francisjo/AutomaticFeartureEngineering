from decimal import Decimal
import pandas as pd
import scipy.stats as stats


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

