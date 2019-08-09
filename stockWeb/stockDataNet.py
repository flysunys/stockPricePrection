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

table_name="netdatastock"
if DBDataNet.exit_of_table(table_name):
	print("存在此表")
else:
	print("不存在此表")
	DBDataNet.establishTable()

#print([[1,1],[2,2],[3,3]])
#df = ts.shibor_ma_data()
stockcode='603993'
start_str='2019-01-01'
end_str='2019-01-12'
starttime=datetime.datetime.strptime(start_str,'%Y-%m-%d')
endtime=datetime.datetime.strptime(end_str,'%Y-%m-%d')
df = ts.get_hist_data('603993',start='2019-01-01',end='2019-01-12')
date_list = []
date_list.append(starttime)
#print(df)
while starttime < endtime:
	starttime+=datetime.timedelta(days=+1)
	date_list.append(starttime)
print(date_list)
for each_day in date_list:
	if DBDataNet.query_of_table(each_day,stockcode):
		print("exit each_day in stockDB")
	else:
		print("not exit each_day in stockDB")
		stockdate=each_day
		stockcode=stockcode
		stockhigh=df['high'].values[0]
		stocklow=df['low'].values[0]
		stockopen=df['open'].values[0]
		stockclose=df['close'].values[0]
		stockvolume=df['volume'].values[0]
		stockadj=df['p_change'].values[0]
		DBDataNet.insert_of_table(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj)
#print(type(df))
#result_shibors=df.values
#for each_day in df:
#	print(each_day_shibor)
#print(df-np.mean(df,axis=0))
#print(np.mean(df,axis=0))

