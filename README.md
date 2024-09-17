# FTSE Time Series Forecasting
Analyse relation of UK Footsie 100 share index(FTSE) with four time series generally from 1988 to 2023, and forescast performance of FTSE in 2024.

## Contents

K54D.xls: Monthly average of private sector weekly pay.

EAFV.xls: Retail sales index, household goods, all businesses.

K226.xls: Extraction of crude petroleum and natural gas.

JQ2J.xls: The manufacturing and business sector of Great Britain, total turnover and orders.

K54D.py: Check seasonality, and apply seasonal additive decomposition; Apply exponential smoothing with additive or multiplicative trends, which is also called "Holt-Winter".

EAFV.py, K226.py, JQ2J.py: Repeat the same operation as K54D.py on 3 other time series.

ARIMA_K54D.py: Apply ARIMA method on K54D time series.

FTSE.py: Make a multiple regression model of FTSE with 4 time series, and forecast.

## Result
Not satisfied with result, which may due to the lack of relevance between FTSE and given time series. Or the multiple regression of degree n may better explain the relation.
