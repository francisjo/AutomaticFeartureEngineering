import pandas.api.types as ptypes
from data_cleaning import CleaningData as cd

class ClassifyColumns:

    def columns_correlation_classification(df, numeric_list, categoric_list):
        corr = df.corr()
        for value in categoric_list:
            if ptypes.is_string_dtype(df[value]):
                continue
            corr_values = corr[value]
            corr_values = corr_values.drop(value)
            for i in corr_values:
                if i > 0.5 or i < -0.5:
                    categoric_list.remove(value)
                    numeric_list.append(value)
                    break
        return numeric_list, categoric_list

    # Detect categorical and numerical variables by Frequency of unique value in Columns Wih Threshold #
    def columns_frequency_classification(df, targetLabel, threshold=0.2):
        numeric_list = []
        categoric_list = []
        for col in df.columns:
            if col == targetLabel:
                continue
            elif ptypes.is_string_dtype(df[col]):
                cd.fill_null_data(df, col, 'string')
                categoric_list.append(col)
            elif ptypes.is_numeric_dtype(df[col]):
                cd.fill_null_data(df, col, 'numeric')
                col_count = len(df[col].unique())
                percent = col_count / df[col].count()
                if percent < threshold:
                    categoric_list.append(col)
                else:
                    numeric_list.append(col)
        new_numeric_list, new_categoric_list = ClassifyColumns.columns_correlation_classification(df, numeric_list, categoric_list)
        return new_numeric_list, new_categoric_list

    # Detect categorical and numerical variables by Frequency of unique value in Columns Wih Threshold #
    def columns_distribution_classification(df, targetLabel, threshold=0.2):
        numeric_list = []
        categoric_list = []
        for col in df.columns:
            if col == targetLabel:
                continue
            elif ptypes.is_string_dtype(df[col]):
                cd.fill_null_data(df, col, 'string')
                categoric_list.append(col)
            elif ptypes.is_numeric_dtype(df[col]):
                cd.fill_null_data(df, col, 'numeric')
                value_count = df.groupby(col)[col].count()
                col_dist = value_count / df[col].count()
                if (col_dist[col_dist > threshold].count()) >= 1:
                    categoric_list.append(col)
                else:
                    numeric_list.append(col)
        new_numeric_list, new_categoric_list = ClassifyColumns.columns_correlation_classification(df, numeric_list, categoric_list)
        return new_numeric_list, new_categoric_list

