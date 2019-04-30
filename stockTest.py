import pandas_datareader.data as web
import datetime
import pandas as pd
import matplotlib.pyplot as plf
start=datetime.date(2012,1,1)
end=datetime.date.today()
stock=web.DataReader('601009.SS','yahoo',start,end)
#stock.to_csv(r'D:\workfiles\stock\data\njyh_601009.csv',columns=stock.columns,index=True)
print(stock['Close'])
fig=plf.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(stock.index,stock['Close'].values)
