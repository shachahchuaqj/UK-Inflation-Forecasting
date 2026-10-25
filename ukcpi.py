import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

cpi_df = pd.read_excel('ukcpi.xlsx', header=None)
cpi_df = cpi_df.rename(columns={0: "Y", 1: "M", 2: "Inflation"})

mthslst = [dt.datetime(year=cpi_df["Y"][i], month=cpi_df["M"][i], day=1) for i in range(len(cpi_df['M']))]
cpi_df['dt'] = mthslst

fig1, ax1 = plt.subplots(figsize=(16,9))

ax1.plot(cpi_df['dt'], cpi_df['Inflation'], color='navy', lw=0.5, label='Past')
ax1.axhspan(1, 3, color='lightgrey', alpha=0.25)
ax1.grid(which='major', axis='both', ls='--', lw=0.5)
ax1.minorticks_on()
ax1.grid(which='minor', axis='both', ls=':', lw=0.2)
ax1.set_xlabel('Year', fontsize=18)
ax1.set_ylabel('12m Inflation Rate (%)', fontsize=18)
ax1.set_title('12m CPI Inflation Rate in UK, 2004 onwards', fontsize=22)

plt.savefig('inflation.png')