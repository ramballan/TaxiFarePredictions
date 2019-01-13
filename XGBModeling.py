from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from xgboost import XGBRegressor

from DataConvertion import timing


@timing
def xgb_modeling(X, y):
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=5, shuffle=True, test_size=0.2)
    xgb_regressor = XGBRegressor(n_estimators=10000,
                                 learning_rate=0.01,
                                 random_state=99,
                                 max_depth=5,
                                 subsample=1.0,
                                 gamma=0.3,
                                 min_child_weight=4,
                                 colsample_bytree=0.8,
                                 n_jobs=-1)
    xgb_regressor.fit(train_X, train_y, verbose=True, early_stopping_rounds=500, eval_metric='rmse',
                      eval_set=[(val_X, val_y)])
    predictions = xgb_regressor.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    result = sqrt(mean_squared_error(val_y, predictions))
    print('rmse={}'.format(result))
    print('mae={}'.format(mae))
    return xgb_regressor
# rmse=3.001048956001278


@timing
def grid_search_xgb(X, y):
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=5, shuffle=True, test_size=0.2)
    param_grid = {'min_child_weight': [4, 5],
                  'gamma': [i / 10.0 for i in range(3, 6)],
                  'subsample': [i / 10.0 for i in range(6, 11)],
                  'colsample_bytree': [i / 10.0 for i in range(6, 11)],
                  'max_depth': [2, 3, 4, 5, 6, 7]}
    xgb_regressor = XGBRegressor(n_estimators=1000,
                                 learning_rate=0.02,
                                 random_state=99,
                                 n_jobs=-1)
    gs = GridSearchCV(xgb_regressor,
                      param_grid=param_grid,
                      verbose=3,
                      cv=TimeSeriesSplit(n_splits=3).get_n_splits([train_X, train_y]))
    gs.fit(X, y, early_stopping_rounds=150, eval_metric='rmse', eval_set=[(val_X, val_y)], verbose=False)
    print(gs.best_params_)
    print(gs.best_score_)
    predictions = gs.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    result = sqrt(mean_squared_error(val_y, predictions))
    print('rmse={}'.format(result))
    print('mae={}'.format(mae))
    return gs

    # rmse = 0.0020213488546884095
    # mae = 0.0004449912412449032
