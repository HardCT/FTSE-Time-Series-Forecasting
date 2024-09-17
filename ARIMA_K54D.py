import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import warnings
import itertools

matplotlib.use('Qt5Agg')
sns.set_style("whitegrid")

# Load excel files and restore as time series. Visualization shows developing changes.
series = pd.read_excel('K54Ddata.xls', header=0, index_col=0).squeeze()
series.index = pd.to_datetime(series.index, format='%Y %b')
series.index = pd.DatetimeIndex(series.index.values, freq=series.index.inferred_freq)
series.plot(title="Private Sector Weekly Pay", xlabel="Year", ylabel="K54D")
pyplot.show()

# Apply seasonally differenced series.
SeasDiff = list()
for i in range(12, len(series)):
    value = series[i] - series[i - 12]
    SeasDiff.append(value)

# Apply first differenced series.
SeasFirstDiff = list()
for i in range(1, len(SeasDiff)):
    value = SeasDiff[i] - SeasDiff[i - 1]
    SeasFirstDiff.append(value)

# Time, ACF, and PACF plots of original data.
fig, ax = pyplot.subplots(3, 1, figsize=(10, 7))
ax[0].plot(series)
ax[0].set_title('Time plot original data')
plot_acf(series, title='ACF plot of original data', lags=50, ax=ax[1])
plot_pacf(series, title='PACF plot of original data', lags=50, ax=ax[2])
pyplot.tight_layout()
pyplot.show()

# Time, ACF, and PACF plots of seasonally differenced series.
fig, ax = pyplot.subplots(3, 1, figsize=(10, 7))
ax[0].plot(SeasDiff)
ax[0].set_title('Time plot seasonally differenced series')
plot_acf(SeasDiff, title='ACF plot of seasonally differenced series', lags=50, ax=ax[1])
plot_pacf(SeasDiff, title='PACF plot of seasonally differenced series', lags=50, ax=ax[2])
pyplot.tight_layout()
pyplot.show()

# Time, ACF, and PACF plots of seasonally and first differenced series.
fig, ax = pyplot.subplots(3, 1, figsize=(10, 7))
ax[0].plot(SeasFirstDiff)
ax[0].set_title('Time plot seasonally + first differenced series')
plot_acf(SeasFirstDiff, title='ACF plot of seasonally + first differenced series', lags=50, ax=ax[1])
plot_pacf(SeasFirstDiff, title='PACF plot of seasonally + first differenced series', lags=50, ax=ax[2])
pyplot.tight_layout()
pyplot.show()

# ADF test of original data.
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# ADF test of seasonally differenced series.
result = adfuller(SeasDiff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# ADF test of seasonally and first differenced series.
result = adfuller(SeasFirstDiff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Explore the best p,d,q value for SARIMAX. Model with the smallest AIC is best option.
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

warnings.filterwarnings("ignore")
best_score, best_param, best_paramSeasonal = float("inf"), None, None
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(series, order=param, seasonal_order=param_seasonal,
                                            enforce_invertibility=False)
            results = mod.fit()
            if results.aic < best_score:
                best_score, best_param, best_paramSeasonal = results.aic, param, param_seasonal
            print('ARIMA{}x{}, with AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
print('The best model is ARIMA{}x{}, with AIC:{}'.format(best_param, best_paramSeasonal, best_score))

# Under this case, ARIMAX(1,1,1)(1,1,1)12 has the best outcome.
mod = sm.tsa.statespace.SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = mod.fit(disp=False)
print(results.summary())

# Summary of graphical results of ARIMA(1,1,1)(1,1,1)12.
results.plot_diagnostics(figsize=(15, 12))
pyplot.show()

# Predicted data of previous model.
pred = results.get_prediction(start=pd.to_datetime('2023-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = series['2000':].plot(label='Original data')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
pyplot.legend()
pyplot.show()

# Predicted data with confidence interval.
pred_uc = results.get_forecast(steps=20)
pred_ci = pred_uc.conf_int()
ax = series.plot(label='Original data')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast values', title='Forecast plot with confidence interval')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
pyplot.legend()
pyplot.show()

# Calculate MSE of forecasting.
y_forecasted = pred.predicted_mean
y_truth = series['2023-01-01':]

MSE = ((y_forecasted - y_truth) ** 2).mean()
print('MSE of the forecasts is {}'.format(round(MSE, 2)))
