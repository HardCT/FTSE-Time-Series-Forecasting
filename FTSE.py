import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
from statsmodels.formula.api import ols
from matplotlib.legend_handler import HandlerLine2D

matplotlib.use('Qt5Agg')
sns.set_style("whitegrid")


# Read four data files and store into time series.
def create_series(file):
    series = pd.read_excel(file, header=0, index_col=0).squeeze()
    series.index = pd.to_datetime(series.index, format='%Y %b')
    series.index = pd.DatetimeIndex(series.index.values, freq=series.index.inferred_freq)
    return series

EAFV = create_series('EAFVdata.xls')
JQ2J = create_series('JQ2Jdata.xls')
K54D = create_series('K54Ddata.xls')
K226 = create_series('K226data.xls')
time_series_dfs = [EAFV, JQ2J, K54D, K226]
start = max(df.index.min() for df in time_series_dfs)
end = min(df.index.max() for df in time_series_dfs)


# Read FTSE file and select time slot that all time series are valid.
FTSE = pd.read_excel('FTSEdata.xls', header=0, index_col=0).squeeze()
FTSE.index = pd.to_datetime(FTSE.index, format='%d/%m/%Y', dayfirst=True)
Selected_FTSE = FTSE.loc[end:start]

# Time plot of FTSE_Open.
Selected_FTSE.reset_index(inplace=True)
merged_df = Selected_FTSE[['Date', 'Open']].copy()
merged_df.rename(columns={'Open': 'FTSE_Open'}, inplace=True)
merged_df.sort_values(by='Date', ascending=True, inplace=True)
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df.sort_values(by='Date', ascending=True, inplace=True)
merged_df.set_index('Date', inplace=True)
merged_df['FTSE_Open'].plot(title="FTSE 100 Financial Times Index", xlabel="Year", ylabel="FTSE_Open")
pyplot.show()

# ADF test of FTSE_Open to check stationarity.
result = adfuller(merged_df['FTSE_Open'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Use seasonal decomposition to analyse data components.
result = seasonal_decompose(merged_df['FTSE_Open'], model='additive')
result.plot()
pyplot.show()

# Detect outliers of DFTSE_OPEN, for rapid increase or decrease.
merged_df['DFTSE_Open'] = merged_df['FTSE_Open'].diff(periods=1)
q1 = merged_df['DFTSE_Open'].quantile(0.25)
q3 = merged_df['DFTSE_Open'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - (0.9 * iqr)
upper_bound = q3 + (0.9 * iqr)
outliers = merged_df[(merged_df['DFTSE_Open'] < lower_bound) | (merged_df['DFTSE_Open'] > upper_bound)]
outliers.rename(columns={'DFTSE_Open': 'DOutlier'}, inplace=True)
pyplot.scatter(outliers.index, outliers['DOutlier'])
pyplot.title("Outliers")
pyplot.xlabel("Year")
pyplot.ylabel("DFTSE_Open")
pyplot.show()

# Merge all files to be a completed form.
merged_df = merged_df.merge(outliers['DOutlier'], how='left', left_on='Date', right_on=outliers['DOutlier'].index)
merged_df['DOutlier'].fillna(0, inplace=True)
for df in time_series_dfs:
    ts = df[(df.index >= start) & (df.index <= end)]
    merged_df = merged_df.merge(ts, how='left', left_on='Date', right_on=ts.index)
merged_df['DJQ2J'] = merged_df['JQ2J'].diff(periods=1)
series = merged_df[(merged_df['Date'] >= '2000-02-01')]

# Reading the basic variables.
DOutlier = series.DOutlier
FTSE_Open = series.FTSE_Open
EAFV = series.EAFV
DJQ2J = series.DJQ2J
JQ2J = series.JQ2J
K54D = series.K54D
K226 = series.K226

warnings.filterwarnings("ignore")

# Forecasting for EAFV using HW-multiplicative seasonality method.
fit1 = ExponentialSmoothing(EAFV, seasonal_periods=12, trend='add', seasonal='mul', damped_trend=True).fit()
fcast1 = fit1.forecast(12).rename("Opt HW-multiplicative seasonality")
fit1.fittedvalues.plot(color='#7BA23F')
fcast1.plot(color='red', legend=True)
EAFV.plot(color='black', legend=True)
pyplot.title('Forecast of EAFV with Holt-Winter method')
pyplot.show()

# Forecasting for JQ2J using HW-additive seasonality method.
fit2 = ExponentialSmoothing(JQ2J, seasonal_periods=12, trend='add', seasonal='add', damped_trend=True).fit()
fcast2 = fit2.forecast(12).rename("Opt HW-additive seasonality")
fit2.fittedvalues.plot(color='#7BA23F')
fcast2.plot(color='red', legend=True)
JQ2J.plot(color='black', legend=True)
pyplot.title('Forecast of JQ2J with Holt-Winter method')
pyplot.show()

# Forecasting for K54D using HW-multiplicative seasonality method.
fit3 = ExponentialSmoothing(K54D, seasonal_periods=12, trend='add', seasonal='mul', damped_trend=True).fit()
fcast3 = fit3.forecast(12).rename("Opt HW-multiplicative seasonality")
fit3.fittedvalues.plot(color='#7BA23F')
fcast3.plot(color='red', legend=True)
K54D.plot(color='black', legend=True)
pyplot.title('Forecast of D3to4 with Holt-Winter method')
pyplot.show()

# Forecasting for K226 using HW-multiplicative seasonality method.
fit4 = ExponentialSmoothing(K226, seasonal_periods=12, trend='add', seasonal='mul', damped_trend=True).fit()
fcast4 = fit4.forecast(12).rename("Opt HW-multiplicative seasonality")
fit4.fittedvalues.plot(color='#7BA23F')
fcast4.plot(color='#7BA23F', legend=True)
K226.plot(color='black', legend=True)
pyplot.title('Forecast of K226 with Holt-Winter method')
pyplot.show()

# Forecasting for DOutlier using Holt's linear method.
fit5 = Holt(DOutlier).fit(optimized=True)
fcast5 = fit5.forecast(12).rename("Holt's linear method")
fit5.fittedvalues.plot(color='#7BA23F')
fcast5.plot(color='#7BA23F', legend=True)
DOutlier.plot(color='black', legend=True)
pyplot.title("Forecast of DOutlier with Holt's linear method")
pyplot.show()

# Forecasting for DJQ2J using HW-additive seasonality method.
fit6 = ExponentialSmoothing(DJQ2J, seasonal_periods=12, trend='add', seasonal='add', damped_trend=True).fit()
fcast6 = fit6.forecast(12).rename("Opt HW-additive seasonality")
fit6.fittedvalues.plot(color='#7BA23F')
fcast6.plot(color='#7BA23F', legend=True)
DJQ2J.plot(color='black', legend=True)
pyplot.title('Forecast of DJQ2J with Holt-Winter method')
pyplot.show()

# Regression model of Ordinary Least Squares(OLS).
formula = 'FTSE_Open ~ EAFV + JQ2J + K54D + K226 + DJQ2J + DOutlier'
results = ols(formula, data=series).fit()
print(results.summary())

b0 = results.params.Intercept
b1 = results.params.EAFV
b2 = results.params.JQ2J
b3 = results.params.K54D
b4 = results.params.K226
b5 = results.params.DOutlier
b6 = results.params.DJQ2J

# Arrays of the fitted values of EAFV, JQ2J, K54D, K226, DJQ2J and DOutlier.
a1 = np.array(fit1.fittedvalues)
a2 = np.array(fit2.fittedvalues)
a3 = np.array(fit3.fittedvalues)
a4 = np.array(fit4.fittedvalues)
a5 = np.array(fit5.fittedvalues)
a6 = np.array(fit6.fittedvalues)

# The fitted part of the forecast of FTSE.
F = a1
for i in range(len(a1)):
    F[i] = b0 + a1[i] * b1 + a2[i] * b2 + a3[i] * b3 + a4[i] * b4 + a5[i] * b5 + a6[i] * b6

# Putting the values of the forecasts of EAFV, JQ2J, K54D, K226, DJQ2J and DOutlier in arrays.
v1 = np.array(fcast1)
v2 = np.array(fcast2)
v3 = np.array(fcast3)
v4 = np.array(fcast4)
v5 = np.array(fcast5)
v6 = np.array(fcast6)

# Building the 12 values of the forecast(for whole year of 2024).
E = v1
for i in range(12):
    E[i] = b0 + v1[i] * b1 + v2[i] * b2 + v3[i] * b3 + v4[i] * b4 + v5[i] * b5 + v6[i] * b6

# Joining the fitted values of the forecast and the points ahead.
K = np.append(F, E)


# Evaluating the MSE.
Error = FTSE_Open - F
MSE = sum(Error ** 2) * 1.0 / len(F)
print(MSE)

# Plotting the graphs of K and FTSE with legends.
line1, = pyplot.plot(K, color='#7BA23F', label='Forecast values')
line2, = pyplot.plot(FTSE_Open, color='black', label='Original data')
pyplot.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
pyplot.title('FTSE regression forecast with confidence interval')
pyplot.show()
