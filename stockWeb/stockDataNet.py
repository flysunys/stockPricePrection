#coding:utf8

import tushare as ts
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plf
from matplotlib.pyplot import MultipleLocator
#from os import path as pth
#p=pth.split(pth.realpath('G:\gitdir\gitprojects\stockPricePrection\stockWeb\DButils\netdatastock.py'))[0]
#pth.sys.path.append(p)
import sys
sys.path.append(r'G:\gitdir\gitprojects\stockPricePrection\stockWeb\DButils')
from netdatastock import DBDataNet

def insertalltotable(stock,stock_code):
	#print(stock['close'].values[1:])
	labelnum_a=stock['close'].values
	labelnum_b=labelnum_a[:-1]
	labelnum_c=labelnum_b.tolist()
	labelnum_c.insert(0,3.3)
	#print(labelnum_c)
	#print(type(labelnum_b.tolist()))
	#labelnum_c=[0,labelnum_b]
	for x1,x2,x3,x4,x5,x6,x7,x8 in zip(stock.index,stock['high'].values,stock['low'].values,stock['open'].values,stock['close'].values,stock['volume'].values,stock['p_change'].values,labelnum_c):
		if DBDataNet.query_of_table(x1,stock_code):
			print("exit each_day in stockDB")
		else:
			print("not exit each_day in stockDB")
			stockdate=x1
			stockcode=stock_code
			stockhigh=float(round(x2,2))
			stocklow=float(round(x3,2))
			stockopen=float(round(x4,2))
			stockclose=float(round(x5,2))
			stockvolume=float(round(x6,2))
			stockadj=float(round(x7,2))
			label=x8
			print(type(x8))
			#print(type(stockdate))
			#print(type(stockhigh))
			#print(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj)
			DBDataNet.insert_of_table(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj,label)

table_name="netdatastock"
if DBDataNet.exit_of_table(table_name):
	print("存在此表")
else:
	print("不存在此表")
	DBDataNet.establishTable()

#print([[1,1],[2,2],[3,3]])
#df = ts.shibor_ma_data()
stockcode='601668'
start_str='2017-06-08'
end_str='2019-08-09'
starttime=datetime.datetime.strptime(start_str,'%Y-%m-%d')
endtime=datetime.datetime.strptime(end_str,'%Y-%m-%d')
df = ts.get_hist_data(stockcode,start=start_str,end=end_str)
date_list = []
date_list.append(starttime)
#temp=df.index
#print(df.index[:-1])
while starttime < endtime:
	starttime+=datetime.timedelta(days=+1)
	date_list.append(starttime)
#print(date_list)
insertalltotable(df,stockcode)
close_price_list=DBDataNet.query_for_table("StockClose")
stock_date_list=DBDataNet.query_for_table("StockDate")
#print(close_price_list[:-1])
#print(type(1.1))
#for each_price, each_date in zip(close_price_list[1:],stock_date_list[:-1]):
#	print(type(each_price[0]),type(each_date[0]))
	#DBDataNet.update_for_table("Label",each_price[0],each_date[0],stockcode)
#print(type(df))
#result_shibors=df.values
#for each_day in df:
#	print(each_day_shibor)
#print(df-np.mean(df,axis=0))
#print(np.mean(df,axis=0))

