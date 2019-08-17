#coding:utf8

import tushare as ts
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plf
from matplotlib.pyplot import MultipleLocator
import sys
sys.path.append(r'G:\gitdir\gitprojects\stockPricePrection\stockWeb\DButils')
from netdatastock import DBDataNet

def normalize_data(data):
	mean_data=np.mean(data)
	max_data=max(data)
	min_data=min(data)
	return [(float(i)-mean_data)/(max_data-min_data) for i in data]

	
def leakrelu(x,a):
	return np.maximum(0,x)+(x<0)*a*x

def leakrelu_gradient(x,a):
	return 1*(x>0)+(x<0)*a	
	
def relu(x):
	return np.maximum(0,x)
	
def relu_gradient(x):
	return 1*(x>0)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
	return x*(1-x)

def prediction(inputs,weights):
	inputs=inputs.astype(float)
	#outputs=sigmoid(np.dot(inputs,weights))
	#outputs=relu(np.dot(inputs,weights))
	outputs=leakrelu(np.dot(inputs,weights),0.0001)
	return outputs

def train(train_inputs,train_outputs,weights,iterations,alpha):
	for iter in range(iterations):
		outputs=prediction(train_inputs,weights)
		error=train_outputs-outputs
		#print(error)
		#print(outputs)
		#adjusts=np.dot(train_inputs.T,alpha*error*sigmoid_derivative(outputs))
		#adjusts=np.dot(train_inputs.T,alpha*error*relu_gradient(outputs))
		adjusts=np.dot(train_inputs.T,alpha*error*leakrelu_gradient(outputs,0.0001))
		print(adjusts)
		weights+=adjusts
	return weights

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
training_output=[]
training_output.append(result_list)
training_outputs = np.array(training_output).T
np.random.seed(1)
weights = 0.0002 * np.random.random((6,1))
iterations=10000
alpha=0.00000001
#print(training_inputs)
#print(np.shape(training_inputs))
#print(np.shape(weights))
#print(np.shape(training_outputs))
weights_res=train(training_inputs,training_outputs,weights,iterations,alpha)
#print(weights_res)
#test_predic=prediction(training_inputs,weights)
#test_predic=np.dot(training_inputs,weights)
#print(test_predic)
#print(weights)
#print(weights_res[0][0])
#weight_d=[]
#weight_dev=[weights_res[0][0],weights_res[1][0],weights_res[2][0],weights_res[3][0],weights_res[4][0],weights_res[5][0]]
#weight_d.append(weight_dev)
#print(np.array(weight_d))


input_num=[]
input_nums=[]
input_num.append(recordData[-1][2])
input_num.append(recordData[-1][3])
input_num.append(recordData[-1][4])
input_num.append(recordData[-1][5])
input_num.append(recordData[-1][6])
input_num.append(recordData[-1][7])
input_nums.append(input_num)
input_num=np.array(input_num)
#print(np.shape(input_nums))
result_predic=np.dot(input_nums,weights_res)
print(training_outputs.size)
print(result_predic)
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

