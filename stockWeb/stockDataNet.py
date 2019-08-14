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

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
	return x*(1-x)

def prediction(inputs,weights):
	inputs=inputs.astype(float)
	outputs=sigmoid(np.dot(inputs,weights))
	return outputs

def train(train_inputs,train_outputs,weights,iterations):
	for iter in range(iterations):
		outputs=prediction(train_inputs,weights)
		error=train_outputs-outputs
		adjusts=np.dot(train_inputs.T,np.dot(error,sigmoid_derivative(outputs)))
		weights+=adjusts
	return weights

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
#insertalltotable(df,stockcode)
#close_price_list=DBDataNet.query_for_table("StockClose")
#stock_date_list=DBDataNet.query_for_table("StockDate")
recordData=DBDataNet.query_table_code(603993)
#print(recordData)
record_list=[]
result_list=[]
for each_record_data in recordData[:-1]:
	high_data=each_record_data[2]
	low_data=each_record_data[3]
	open_data=each_record_data[4]
	close_data=each_record_data[5]
	volume_data=each_record_data[6]
	adj_data=each_record_data[7]
	temp_list=[]
	temp_list.append(high_data)
	temp_list.append(low_data)
	temp_list.append(open_data)
	temp_list.append(close_data)
	temp_list.append(volume_data)
	temp_list.append(adj_data)
	record_list.append(temp_list)
	label_data=each_record_data[8]
	result_list.append(label_data)
training_inputs = np.array(record_list)
training_outputs = np.array(result_list).T
np.random.seed(1)
weights = 2 * np.random.random((6,1)) - 1
#print(training_inputs)
#print(training_outputs)
iterations=1000
weights_res=train(training_inputs,training_outputs,weights,iterations)
#print(weights_res[0][0])
#weight_d=[]
weight_dev=[weights_res[0][0],weights_res[1][0],weights_res[2][0],weights_res[3][0],weights_res[4][0],weights_res[5][0]]
#weight_d.append(weight_dev)
#print(np.array(weight_d))


input_num=[]
input_num.append(recordData[-1][2])
input_num.append(recordData[-1][3])
input_num.append(recordData[-1][4])
input_num.append(recordData[-1][5])
input_num.append(recordData[-1][6])
input_num.append(recordData[-1][7])
input_num=np.array(input_num)
print(input_num)
result_predic=np.dot(input_num,np.array(weight_dev).T)
print(result_predic.sum())
#print(np.dot())
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

