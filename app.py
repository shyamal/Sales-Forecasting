import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import streamlit as st

data = pd.read_csv('store_demand.csv', parse_dates=['date'])
data.sort_values(by=['date'], inplace=True)
data = data.groupby(['date'])['sales'].sum()


# store_item = data[(data['store'] == 1) & (data['item'] == 1)]
# plt.plot(store_item['date'], store_item['sales'])
# plt.title('Sales Trend for Store 1, Item 1')
# plt.show()

model = ARIMA(data, order=(1, 1, 1))
results = model.fit()
print(results.summary())

st.title ("Saleas Forcasting app")

st.write("## Historical Sales Data")
dataframe = pd.DataFrame(data).reset_index()
st.line_chart(dataframe.rename(columns={'date' : 'index'}).set_index('index'))

# predictions = results.forecast(steps=30)
# actuals = store_item['sales'][-30:]  # Adjust this slice to match your forecasting range
# rmse = np.sqrt(mean_squared_error(actuals, predictions))
# print(f"RMSE: {rmse}")

st.write('### Forcast Future Sales')
steps = st.number_input('Enter number of days to forecast:', min_value=1, max_value=365, value=30)

if st.button('Forecast'):
    predictions = results.forecast(steps=steps)
    st.write(f"Forecast for the next {steps} days:")
    st.line_chart(predictions)