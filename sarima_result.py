from ukcpi import cpi_df
import datetime as dt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

#with optimised model SARIMA(1,1,1)(0,0,1,12):
model = SARIMAX(cpi_df['Inflation'], order=(1, 1, 1), seasonal_order = (0, 0, 1, 12))
model_fit = model.fit(disp=False, method='lbfgs')

#now to make predictions on the UK inflation rate up to May 2029 (another 37 months):
nummths = 37
N = len(cpi_df['Inflation'])
forecast = model_fit.get_forecast(steps=nummths)
mean_forecast = forecast.predicted_mean
conf_int = np.transpose(np.array(forecast.conf_int(alpha=0.10)))
stderr = forecast.summary_frame()['mean_se']

cpi_pred = []
i, yr, mth = 0, cpi_df['Y'][N - 1], cpi_df['M'][N - 1]
while i < nummths:
    if mth == 12:
        mth = 1
        yr += 1
    else:
        mth += 1
    
    cpi_pred.append(str(dt.datetime(year=yr, month=mth, day=1)))
    
    i += 1

cpi_pred = pd.DataFrame({'dt': cpi_pred, 
                         'Inflation': mean_forecast,
                         '5%': conf_int[0],
                         '95%': conf_int[1],
                         'stderr': stderr}, 
                         columns=['dt','Inflation','5%','95%','stderr'])

cpi_pred.to_excel("SARIMA_forecast.xlsx")