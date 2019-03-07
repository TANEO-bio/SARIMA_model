#!/usr/bin/env python3

import pandas_datareader.data as pdr
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels
import statsmodels.api as sm
import numpy as np
import sys
import random
import string
from datetime import datetime as dt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

coad = sys.argv[1]
day = int(sys.argv[2])
mpl.use('Agg')
letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
randstr = ''.join([random.choice(letters) for _ in range(12)])
# ----Collecting Data----

end = datetime.date.today()
start = end - datetime.timedelta(days=1200)

df = pdr.DataReader('MSFT', 'iex', start, end)

df = df['close']
df = df[len(df)-800-day:]

df.plot(grid=True)
plt.tick_params(labelsize=7)
plt.savefig(str(randstr)+'.png')
randstr = ''.join([random.choice(letters) for _ in range(12)])
plt.figure()

# ----Auto Correlation----

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
sm.graphics.tsa.plot_acf(df, lags=[i for i in range(1, 15)], ax=ax1)
sm.graphics.tsa.plot_pacf(df, lags=[i for i in range(1, 15)], ax=ax2)
plt.savefig(str(randstr)+".png")
randstr = ''.join([random.choice(letters) for _ in range(12)])
plt.figure()

# ----seasonal decompose----

res = statsmodels.tsa.seasonal.seasonal_decompose(df, freq=5)

fig,axes = plt.subplots(nrows=4, ncols=1,figsize=(12,12))

axes[0].plot(res.observed.index, res.observed)
axes[0].set_title('Original')
axes[0].grid(True)

axes[1].plot(res.trend.index, res.trend)
axes[1].set_title('Trend')
axes[1].grid(True)

axes[2].plot(res.seasonal.index, res.seasonal)
axes[2].set_title('Seasonal')
axes[2].grid(True)
axes[2].ticklabel_format(style="sci",  axis="y")

axes[3].plot(res.resid.index, res.resid)
axes[3].set_title('Resid')
axes[3].grid(True)

plt.savefig(str(randstr)+".png")
randstr = ''.join([random.choice(letters) for _ in range(12)])
plt.figure()

# ----Cross Validation----

best_param_list = []
for epoch in range(0, 4):
    train = df[0+epoch*200:200-day+epoch*200]
    test = df[200-day+epoch*200:200+epoch*200]
    diff = train - train.shift()
    diff = diff.dropna()
    param = sm.tsa.arma_order_select_ic(diff, ic='aic', trend='nc')
    param = param["aic_min_order"]
    param_list = []
    rmse_list = []
    print(param)

# ARMA model
    order = (param[0], 0, param[1])
    seasonal_order = (0, 0, 0, 0)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit()
    pred = result.forecast(day)
    rmse = np.sqrt(mean_squared_error(test, pred))
    print(rmse)
    param_list.append([order, seasonal_order])
    rmse_list.append(rmse)

# ARIMA model
    order = (param[0], 1, param[1])
    seasonal_order = (0, 0, 0, 0)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit()
    pred = result.forecast(day)
    rmse = np.sqrt(mean_squared_error(test, pred))
    print(rmse)
    param_list.append([order, seasonal_order])
    rmse_list.append(rmse)

# SARIMA model
    order = (param[0], 1, param[1])
    seasonal_order = (1, 1, 1, 5)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit()
    pred = result.forecast(day)
    rmse = np.sqrt(mean_squared_error(test, pred))
    print(rmse)
    param_list.append([order, seasonal_order])
    rmse_list.append(rmse)

# Comparison
    best_param = param_list[rmse_list.index(min(rmse_list))]
    best_param_list.append(best_param)

# ----Model Selection----

ms_param_list = []
ms_rmse_list = []
result_list = []
for validation in best_param_list:
    train = df[0:800]
    test = df[800:800+day]
    model = SARIMAX(train, order=validation[0], seasonal_order=validation[1], enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit()
    pred = result.forecast(day)
    rmse = np.sqrt(mean_squared_error(test, pred))
    ms_param_list.append([validation[0], validation[1]])
    ms_rmse_list.append(rmse)
    
    train_pred = result.predict()
    test_pred = pred
    result_list.append([train_pred, test_pred])

best_result = result_list[ms_rmse_list.index(min(ms_rmse_list))]
best_parameter = ms_param_list[ms_rmse_list.index(min(ms_rmse_list))]
train_pred = best_result[0].values.tolist()
test_pred = best_result[1]

# ----Describing Results----


plt.plot(df[795:800+day].index, df[795:800+day], linestyle='dashed', color='blue', label='Original')
plt.plot(df[795:800].index, train_pred[795:800], linestyle='solid', color='green', label='Predict(train)')
pred_list = test_pred.tolist()
print("this")
print(pred_list)
print(pred_list[len(pred_list[795:800+day])-1])
pred_list.insert(0, train_pred[len(train_pred)-1])
plt.plot(df[799:800+day].index, pred_list, linestyle='solid', color='red', label='Predict(test)')
plt.tick_params(labelsize=6)
plt.legend()
plt.savefig(str(randstr)+".png")
randstr = ''.join([random.choice(letters) for _ in range(12)])
plt.figure()

model = SARIMAX(df, order=best_parameter[0], seasonal_order=best_parameter[1], enforce_stationarity=False, enforce_invertibility=False)
result = model.fit()
pred = result.forecast(day)
datelist = []
start = end
date = df[795:800+day].index.tolist()
for i in range(0, day+1):
    strdt = dt.strptime(date[len(df[795:800+day])-1], '%Y-%m-%d')
    day_n = strdt + datetime.timedelta(days=i)
    datelist.append(day_n.strftime("%Y-%m-%d"))
pred = pred.values.tolist()
pred.insert(0, df[799+day])

plt.plot(df[795:800+day].index, df[795:800+day], color='red', label='Original', linestyle='solid')
plt.plot(datelist, pred, linestyle='dashed', color='blue', label='Forecast')
plt.tick_params(labelsize=6)
plt.legend()
plt.savefig(str(randstr)+".png")

print("end")

