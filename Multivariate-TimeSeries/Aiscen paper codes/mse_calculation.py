import pandas as pd
import numpy as np
df_mse = pd.read_csv('mse_aofa_lond_in.csv')
df_mse = df_mse.set_index('Days')
y_forecasted = df_mse['y_forecasted']
y_truth = df_mse['y_truth']
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))