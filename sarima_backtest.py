#backtest with parameters p=1, q=1, P=0, D=0, Q=1, together with fixed d=1, s=12.
#using last 24 monnths as training backtest

from ukcpi import cpi_df
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

N = len(cpi_df['Inflation'])
testmths = 24

train = cpi_df['Inflation'][:N-testmths]
test = cpi_df['Inflation'][N-testmths:]

#rolling 1-step forecast
history = list(train)
predictions = []
for t in range(len(test)):

    model = SARIMAX(
        history,
        order=(1,1,1),
        seasonal_order=(0,0,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(
        disp=False,
        method='lbfgs',
        maxiter=100
    )

    forecast = model_fit.get_forecast(steps=1)

    yhat = forecast.predicted_mean
    predictions.extend(yhat)

    history.append(test.iloc[t])

#rms error and mean absolute errors
rmse = np.sqrt(mean_squared_error(test, predictions))
mae = mean_absolute_error(test, predictions)

print("RMSE:", rmse)
print("MAE:", mae)

plt.figure(figsize=(10,5))

plt.plot(cpi_df['dt'][N-testmths:], test, label="Actual")
plt.plot(cpi_df['dt'][N-testmths:], predictions, label="Predicted")

plt.legend()
plt.title("Backtest Results")
plt.savefig('sarima_backtest_res.png')