import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

matplotlib.use('Qt5Agg')
sns.set_style("whitegrid")

# ----------------------------data preparation and preliminary analysis-------------------------------------

# Load excel files and restore as time series. Visualization shows developing changes.
series = pd.read_excel('K226data.xls', header=0, index_col=0).squeeze()
series.index = pd.to_datetime(series.index, format='%Y %b')
series.index = pd.DatetimeIndex(series.index.values, freq=series.index.inferred_freq)
series.plot(title="Extraction of Crude Petroleum and Natural Gas", xlabel="Year", ylabel="K226")
pyplot.show()

# Create seasonal plot to check seasonality.
seasonal_data = pd.DataFrame(index=np.unique(series.index.month), columns=np.unique(series.index.year))
for year in np.unique(series.index.year):
    seasonal_data.loc[:np.sum(series.index.year == year), year] = series[series.index.year == year].values
seasonal_data.plot(title="Extraction of Crude Petroleum and Natural Gas", xlabel="Month", ylabel="K226", legend=False)
x = np.array(range(1, 13))
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
pyplot.xticks(x, months)
pyplot.show()

# Apply seasonal additive decomposition to determine trends.
result1 = seasonal_decompose(series, model='additive')
result1.plot()
pyplot.show()

# ------------------------implementation of exponential smoothing-------------------------------

# Choose additive or multiplicative trends to apply exponential smoothing, which is also called "Holt-Winter".
fit1 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='add', damped_trend=True).fit()

fit2 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='mul', damped_trend=True).fit()

# Calculate mean squared error to compare accuracy.
MSE1 = mean_squared_error(fit1.fittedvalues, series)
MSE2 = mean_squared_error(fit2.fittedvalues, series)

# Show details of 4 models, including coefficients and MSE.
results = pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"10", "b0", "MSE"])
params = ['smoothing_level', 'smoothing_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
results["HW model 1"] = [fit1.params[p] for p in params] + [MSE1]
results["HW model 2"] = [fit2.params[p] for p in params] + [MSE2]
print(results)

# Show residuals of 4 models. Equally random residuals would be good.
residuals1 = fit1.fittedvalues - series
residuals2 = fit2.fittedvalues - series
residuals1.rename('residual plot for model 1').plot(color='#66BAB7', legend=True)
residuals2.rename('residual plot for model 2').plot(color='#7BA23F', legend=True)
pyplot.title('Residual plots for models 1 and 2')
pyplot.show()

# Show autocorrelation of 2 models. Good ACF should not show clear trends.
plot_acf(residuals1, title='Residual ACF for model 1', lags=50)
plot_acf(residuals2, title='Residual ACF for model 2', lags=50)
pyplot.show()

# In this case, Holt-Winter with multiplicative seasonality have the best behaviour.
fit2.fittedvalues.plot(color='#7BA23F')
series.rename('Time plot of original series').plot(color='black', legend=True)
fit2.forecast(12).rename('Model 2: Opt HW-multiplicative seasonality').plot(color='#7BA23F', legend=True)

pyplot.xlabel('Dates')
pyplot.ylabel('Values')
pyplot.title('HW method-based forecasts for K226')
pyplot.show()
