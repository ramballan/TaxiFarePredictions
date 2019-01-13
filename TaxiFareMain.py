import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from DataConvertion import convert_dataframe, get_rid_of_outliers
from XGBModeling import grid_search_xgb, xgb_modeling

nrows = 100000
# download data from
# https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data
train_data = pd.read_csv('./data/train.csv', nrows=nrows)
test_data = pd.read_csv('./data/test.csv')

filter_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
train_data = get_rid_of_outliers(train_data, filter_columns)
train_data.info(memory_usage='deep')
convert_dataframe(train_data)
convert_dataframe(test_data)

# ax = train_data.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', color='blue', s=.02, alpha=.1)
# test_data.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', color='green', s=.04, ax=ax)
# plt.show()

X = train_data.select_dtypes(exclude=['object', 'datetime64']).drop(['fare_amount'], axis=1)
y = train_data.fare_amount
test_X = test_data.select_dtypes(exclude=['object', 'datetime64'])

print(X.describe())
print(test_X.describe())

xgb_model = xgb_modeling(X, y)
xgb_predictions = xgb_model.predict(test_X)

results = pd.DataFrame({'key': test_data.key,
                        'fare_amount': xgb_predictions})
results.to_csv('./data/submission.csv', index=False)
