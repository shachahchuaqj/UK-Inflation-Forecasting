from ukcpi import cpi_df, ax1
import datetime as dt
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

d, s = 1, 12

#now need to find p, q, P, D, Q by looking at each of the BIC values.
#D can be limited to just 0 or 1.
#P, Q can also be limited to just 0 or 1 (i.e. only need to consider at most 1 period ago).
#p, q can be limited to 1-3 (i.e. take in information from earliest the last quarter).

combs = list(product([1,2,3],
                    [1,2,3],
                    [0,1],
                    [0,1],
                    [0,1]))

minbic = None
minp = minq = minP = minD = minQ = None
for (p,q,P,D,Q) in combs:
    print(f"currently {p,q,P,D,Q}")                 #??? somehow this one line prevents the keyboardInterrupt

    try:
        model = SARIMAX(cpi_df['Inflation'], order=(p, d, q), seasonal_order = (P, D, Q, s))
        model_fit = model.fit(disp=False,
                              method='lbfgs',
                              maxiter=50)           #apparently these help optimise the computations

        curbic = model_fit.bic
        print(f"bic={curbic} for {p,q,P,D,Q}")
        if minbic is None or curbic < minbic:
            minbic = curbic
            minp, minq, minP, minD, minQ = p, q, P, D, Q

    except Exception as e:
        print(f"{e} for {p,q,P,D,Q}")
        continue

print(f"minbic {minbic} achieved with p={minp}, q={minq}, P={minP}, D={minD}, Q={minQ}")
#achieved with (1,1,0,0,1)

#with optimised model:
model = SARIMAX(cpi_df['Inflation'], order=(minp, d, minq), seasonal_order = (minP, minD, minQ, s))
model_fit = model.fit(disp=False, method='lbfgs')           #no need maxiter limit for best model

#now to make predictions on the UK inflation rate up to May 2029 (another 37 months):
nummths = 37
N = len(cpi_df['Inflation'])
forecast = model_fit.get_forecast(steps=nummths)
mean_forecast = forecast.predicted_mean

cpi_pred = []
i, yr, mth = 0, cpi_df['Y'][N - 1], cpi_df['M'][N - 1]
while i < nummths:
    if mth == 12:
        mth = 1
        yr += 1
    else:
        mth += 1
    
    cpi_pred.append(dt.datetime(year=yr, month=mth, day=1))
    
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

plt.savefig('sarima_pred.png')