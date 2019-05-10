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




# ========================  OLD MAIN FUNCTION ========================== #

'''

# drop the target label from a list
def drop_target_label(cols_list, target_label):
    for value in cols_list:
        if value == target_label:
            cols_list.remove(value)
            break
    return cols_list


# the main function to run the pre-processing phase and to fit the model to training data #
def run_model(df):

    # split data-time column if exists
    #cd.split_datetime_col(df)

    # classify columns to categorical and numerical
    summarized_df, numeric_cols, nominal_cols, ordinal_cols = col_classify.get_numeric_nominal_ordinal_cols(df)
    numeric_cols = drop_target_label(numeric_cols, target_label)
    nominal_cols = drop_target_label(nominal_cols, target_label)
    ordinal_cols = drop_target_label(ordinal_cols, target_label)

    simple_imputer = SimpleImputer(strategy="most_frequent", copy=False)

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
            ('imputer', simple_imputer),
            ('LabelEncoder', ce.BinaryEncoder())
        ]
    )

    categorical_transformer_nominal = Pipeline(
        steps=
        [
            ('imputer', simple_imputer),
            ('OneHotEncoder', ce.OneHotEncoder())
        ]
    )

    # combine transformers in one Preprocessor 
    preprocessor = ColumnTransformer(
        transformers=
        [
            ('num', numeric_transformer, numeric_cols),
            ('cat_ordinal', categorical_transformer_ordinal, ordinal_cols),
            ('cat_nominal', categorical_transformer_nominal, nominal_cols)
        ]
    )

    # append classifier to pre-processing pipeline.
    classifier = Pipeline(
        steps=
        [
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))
        ]
    )

    # Features
    X = df.drop(target_label, axis=1)

    # Target Label
    y = df[target_label]

    # Split Data to Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Fit Model
    classifier.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)
    return score

'''
