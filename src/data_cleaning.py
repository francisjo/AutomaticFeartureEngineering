import pandas.api.types as ptypes
import pandas as pd


# fill null values with respect to the column data-type #
def fill_null_data(df, col, col_type):
    if df[col].isnull().sum() > 0 and col_type == 'string':
        df[col].fillna('missing', inplace=True)
    elif df[col].isnull().sum() > 0 and col_type == 'numeric':
        df[col].fillna(df[col].mean(), inplace=True)


# convert Date col to Datetime and split it to Year,Month,Day Columns #
def split_datetime_col(df):
    # df['Date'] = pd.to_datetime(df['Date'])
    for col in df.columns:
        if ptypes.is_datetime64_ns_dtype(df[col]):
            df[col + '_Day'] = pd.DatetimeIndex(df[col]).day
            df[col + '_Month'] = pd.DatetimeIndex(df[col]).month
            df[col + '_Year'] = pd.DatetimeIndex(df[col]).year

