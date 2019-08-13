#!/usr/bin/env python
# coding: utf-8

# 载入数据
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import cufflinks as cf
import matplotlib.pyplot
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

road = './bike.csv'

def get_data(road):
    dat = pd.read_csv(road)
    dat.drop(['casual', 'registered'], axis = 1, inplace = True)
    return dat

df = get_data(road)
data = df[[col for col in df.columns if col not in ['registed', 'casual']]]


# 特征工程
def data_application(data):
    if 'datetime' in list(data.columns):
        data.datetime = pd.to_datetime(data.datetime)
        data['day'] = data.datetime.apply(lambda x: x.day)
        data['year'] = data.datetime.apply(lambda x: x.year)
        data['hour'] = data.datetime.apply(lambda x: x.hour)
        data['minute'] = data.datetime.apply(lambda x: x.minute)
        data['dayofweek'] = data.datetime.apply(lambda x: x.dayofweek)
        data['weekend'] = data.datetime.apply(lambda x: x.dayofweek in [5, 6])
        data.drop('datetime', axis = 1, inplace = True)

# 构建评价标准
def post_pred(y_pred):
    y_pred[y_pred < 0] = 0
    return y_pred

def rmsle(y_true, y_pred, y_pred_only_postive = True):
    if y_pred_only_postive:
        y_pred = post_pred(y_pred)
    diff = np.log(y_pred + 1) - np.log(y_true + 1)
    mean_err = np.square(diff).mean()
    return np.sqrt(mean_err)


# 模型构建

## 拆分数据集
def assing_test_samples(data, test_ratio = 0.3, seed = 1):
    days = data.day.unique()
    np.random.seed(seed)
    np.random.shuffle(days)
    test_day = days[ :int(len(days) * test_ratio)]
    data['is_test'] = data.day.isin(test_day)

def get_x_y(data, target_variable):
    train_values = data.drop(target_variable, axis = 1).values
    target = data[target_variable].values
    return train_values, target
    
def train_test_split(data, target_variable):
    data_application(data)
    assing_test_samples(data)
    df_train = data[data.is_test == False]
    df_test = data[data.is_test == True]
    X_train, y_train = get_x_y(df_train, 'count')
    X_test, y_test = get_x_y(df_test, 'count')
    return X_train, y_train, X_test, y_test


## 拟合与预测
def fit_and_predict(data, model, target_variable):
    X_train, y_train, X_test, y_test = train_test_split(data, target_variable)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred

def count_prediction(data, model, target_variable = 'count'):
    y_test, y_pred = fit_and_predict(data, model, target_variable)
    return rmsle(y_test, y_pred, y_pred_only_postive = True)


## 模型特征重要性的可视化
def get_features(data, target_variable):
    return list(data.drop(target_variable, axis =1).columns)

def importance_features(model, data, target_variable):
    features = get_features(data, target_variable)
    impdf = []
    fscore = model.get_booster().get_fscore()
    maps_name = dict([('f{0}'.format(i), col) for i, col in enumerate(features)])
    
    for ft, score in fscore.items():
        impdf.append({'features': maps_name[ft], 'importance': score})
    impdf = pd.DataFrame(impdf)
    impdf = impdf.sort_values(by = 'importance', ascending = False).reset_index(drop = True)
    impdf['importance'] /= impdf['importance'].sum()
    impdf.index = impdf['features']
    impdf.drop('features', axis = 1, inplace = True)
    return impdf

def draw_importance_features(model, data, target_variable):
    impdf = importance_features(model, data, target_variable)
    impdf.iplot(kind = 'bar')


# 拟合模型
#model = xgb.XGBRegressor(objective = 'reg:squarederror')
#print('xgboost', count_prediction(data, model))
#draw_importance_features(model, data, 'count')

# 使用贝叶斯优化方法进行调参
def objective(space)
    model = xgb.XGBRegressor(objective = 'reg:squarederror',
                             max_depth = int(space['max_depth']),
                             n_estimators = int(space['n_estimators']),
                             subsample = space['subsample'],
                             colsample_bytree = space['colsample_bytree'],
                             learning_rate = space['learning_rate'],
                             reg_alpha = space['reg_alpha'])

    X_train, y_train, X_test, y_test = train_test_split(data, 'count')
    _, registered_pred = fit_and_predict(data, model, 'registered')
    _, casual_pred = fit_and_predict(data, model, 'casual')

    y_pred = registered_pred + casual_pred

    score = rmsle(y_test, y_pred)

    return {'loss': score, 'status': STATUS_OK}

space = {
    'max_depth': hp.quniform('x_max_depth', 2, 20, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 500, 1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
    'reg_alpha': hp.uniform('reg_alpha', 0.1, 1)
}

trials = Trials()
best = fmin(fn = objective,
            space = space,
            max_evals = 15,
            trials = trials,
            algo = tpe.suggest)

print(best)