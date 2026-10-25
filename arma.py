from ukcpi import cpi_df, ax1
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#from augmented Dickey-Fuller test, p = 0.0752 > 0.05 so need to difference the data
cpi_diff = np.diff(cpi_df['Inflation'])
result = adfuller(cpi_diff)
#print('ADF Statistic:', result[0])
#print('p-value:', result[1])

#doing augmented Dickey-Fuller test again, the p-value is now on the order of 10^-6.
#the first difference is stationary, so we can use an ARIMA(p,1,q) model on the original data because d=1 from differencing once.

#now we need to determine the values of p and q. 
#plotting the PACF and ACF graphs:

fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(16,10))

plot_acf(cpi_diff, ax=ax2)
plot_pacf(cpi_diff, ax=ax3)

#plt.show()

#from the PACF plot, there are significant spikes at lags 2,3,4,9, so the p for the AR could be one of these values. 
#from the ACF plot, there is a sharp cutoff after 9 and 12, so similarly the q for the MA could be one of these values.
#upon trying all the combinations of (p,q), p=2 and q=12 returns the lowest BIC value.

#fitting the ARIMA(3,1,12) model:
model = ARIMA(cpi_df['Inflation'], order=(2, 1, 12))
model_fit = model.fit()

#print(model_fit.summary())

#now to make predictions on the UK inflation rate up to May 2029 (another 39 months):
N = len(cpi_df['Inflation'])
forecast = model_fit.get_forecast(steps=40)
mean_forecast = forecast.predicted_mean

cpi_pred = []
i, yr, mth = 0, cpi_df['Y'][N - 1], cpi_df['M'][N - 1]
while i <= 39:
    cpi_pred.append(dt.datetime(year=yr, month=mth, day=1))

    if mth == 12:
        mth = 1
        yr += 1
    else:
        mth += 1

    i += 1

cpi_pred = pd.DataFrame({'dt': cpi_pred, 'Inflation': mean_forecast}, columns=['dt','Inflation'])

#changing the original inflation rate plot to include these predictions:
ax1.plot(cpi_pred['dt'], cpi_pred['Inflation'], color='coral', lw=0.75, label='Predicted')
#confidence interval plotting:
conf_int = forecast.conf_int()
ax1.fill_between(
    cpi_pred['dt'],
    conf_int.iloc[:,0],
    conf_int.iloc[:,1],
    color='coral',
    alpha=0.2
)

plt.show()