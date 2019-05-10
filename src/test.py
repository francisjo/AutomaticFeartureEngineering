import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from decimal import Decimal

titanic = 'C:\\Users\\Joseph Francis\\AutomaticFeartureEngineering\\Datasets\\train.csv'
df = pd.read_csv(titanic)
'''
crosstab = pd.crosstab(df['Cabin'], df["Fare"])
# crosstab = df.Name.groupby(df['Name']).count()
stat, p, dof, expected = chi2_contingency(crosstab)

prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
    print('Dependent')
else:
    print('Independent')
'''


dff = df[['Name', 'Sex', 'Age']].copy()




def columns_correlation_spearmanr(df):
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


result = columns_correlation_spearmanr(dff)

print(result)