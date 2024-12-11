import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('store_demand.csv', parse_dates=['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day_of_week'] = data['date'].dt.day_of_week


store_item = data[(data['store'] == 1) & (data['item'] == 1)]
plt.plot(store_item['date'], store_item['sales'])
plt.title('Sales Trend for Store 1, Item 1')
plt.show()