import pandas_datareader.data as web
import datetime
import pandas as pd

#the first step is set parameters and get data from datasource.
start = datetime.datetime(2018, 7, 1)

#end = datetime.datetime(2013, 1, 27)
end = datetime.date.today()
#f = web.DataReader('F', 'google', start, end)

stock = web.DataReader("601009.SS", "yahoo", start, end)
print("stock price first 5 days")
print(stock.head(5))

print("stock price last 5 days")
print(stock.tail(5))

#dataset save to excel table

stock.to_csv(r'F:\linux\stockPricePredictionSystem\stockPricePrection\stockDataset\nanjingyinhang601009.csv',columns=stock.columns,index=True)

#the second step is get figure from dataset

print("the close price figure")

print(stock.shape)
print(stock.info)
print(stock.describe())


#the third step is analysis data
change = stock.Close.diff()
stock['Change'] = change
print(stock.head(5))
# change.fillna(change.mean(),inplace=True)
# stock['pct_change'] = (stock ['Change'] / stock ['Close'].shift(1))
# stock['pct_change1'] = stock .Close.pct_change()
# jump_pd = pd.DataFrame()
# for kl_index in np.arange(1, stock.shape[0]):
  # today = stock.ix[kl_index]
  # yesday = stock.ix[kl_index-1]
  # today['preCloae'] = yesday.Close
            
  # if today['pct_change'] > 0 and (today.Low-today['preCloae']) > 0:
    # today['jump_power'] = (today.Low-today['preCloae'])
  # elif  today['pct_change'] < 0 and (today.High-today['preCloae']) < 0:
    # today['jump_power'] = (today.High-today['preCloae'])
    # jump_pd = jump_pd.append(today)        
  # stock['jump_power'] = jump_pd['jump_power']
  # print stock.loc["2017-04-26":"2017-06-15"]
# format = lambda x: '%.2f' % x
# stock = stock.applymap(format)
# print stock.loc["2017-04-26":"2017-06-15"]