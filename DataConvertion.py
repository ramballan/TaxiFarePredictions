import time
from math import sqrt

import pandas as pd


def timing(f):
    def wrap(*args):
        print('{:s} function started'.format(f.__name__))
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.2f} sec'.format(f.__name__, (time2 - time1)))

        return ret

    return wrap


@timing
def convert_dataframe(df):
    df.dropna(axis=0, inplace=True);
    df['pickup_datetime_dt'] = pd.to_datetime(df.pickup_datetime)
    df['year'] = df['pickup_datetime_dt'].dt.year
    # df['year'] = df['year'] - df['year'].min()
    df['month'] = df['pickup_datetime_dt'].dt.month
    df['day'] = df['pickup_datetime_dt'].dt.day
    df['hour'] = df['pickup_datetime_dt'].dt.hour
    df['day_of_week'] = df['pickup_datetime_dt'].dt.dayofweek
    df['distance'] = ((df.pickup_longitude - df.dropoff_longitude) ** 2 + (
                df.pickup_latitude - df.dropoff_latitude) ** 2).apply(lambda x: sqrt(x))
    return df


@timing
def get_rid_of_outliers(df, columns):
    for column in columns:
        df = df.loc[pd.np.abs(df[column] - df[column].mean()) <= (3 * df[column].std())]
        df = df.loc[~(pd.np.abs(df[column] - df[column].mean()) > (3 * df[column].std()))]
    return df
