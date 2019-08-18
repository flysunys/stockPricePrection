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

#函数定义

def array_oneTotwo(x_array):
	t=np.empty((x_array.shape[0],1))
	for i in range(x_array.shape[0]):
		t[i,:]=x_array[i]
	return t
	
#def array_twoToone(x_array):
#	t=np.empty((max(x_array.shape),))
#	for i in range(x_array.shape[0]):
#		t[i]=x_array[i]
#	return t

def maxminnorm(array):
	maxcols=array.max(axis=0)
	mincols=array.min(axis=0)
	data_shape = array.shape
	data_rows = data_shape[0]
	data_cols = data_shape[1]
	t=np.empty((data_rows,data_cols))
	for i in range(data_cols):
		t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
	return t

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

def prediction(input_t,weight_t):
	input_t=input_t.astype(float)
	#outputs=sigmoid(np.dot(inputs,weights))
	#outputs=relu(np.dot(inputs,weights))
	output_t=leakrelu(np.dot(input_t,weight_t),0.0001)
	return output_t

def train(train_inputs,train_outputs,weights,iterations,alpha):
	for iter in range(iterations):
		#forwad propagation
		outputs_one=prediction(train_inputs,weights[0])
		#print(outputs_one)
		outputs_two=prediction(outputs_one,weights[1])
		print(outputs_two)
		#backward propagation
		error_two=outputs_two-train_outputs
		weights_derivative_two=np.dot(outputs_one.T,alpha[0]*error_two)/train_outputs.size
		error_one=np.dot(error_two,weights[1].T)*leakrelu_gradient(np.dot(train_inputs,weights[0]),0.0001)
		weights_derivative_one=np.dot(train_inputs.T,alpha[1]*error_one)/train_outputs.size
		print("weights_derivative_two: %s" % weights_derivative_two)
		print("weights_derivative_one: %s" % weights_derivative_one)
		#print(outputs)
		#adjusts=np.dot(train_inputs.T,alpha*error*sigmoid_derivative(outputs))
		#adjusts=np.dot(train_inputs.T,alpha*error*relu_gradient(outputs))
		#adjusts=np.dot(train_inputs.T,alpha*error*leakrelu_gradient(outputs,0.0001))
		#print(adjusts)
		weights[0]-=weights_derivative_one
		weights[1]-=weights_derivative_two
	return weights

	
#数据获取
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
training_inputs_d = np.array(record_list)

# normalize inputs

training_inputs=maxminnorm(training_inputs_d)
#训练集
train_data_input=training_inputs[0:401,:]
#验证集
valid_data_input=training_inputs[401:-1,:]

training_output=[]
training_output.append(result_list)
training_outputs_d = np.array(training_output).T

# normalize outputs

training_outputs=maxminnorm(training_outputs_d)
#训练集
train_data_output=training_outputs[0:401,:]
#验证集
valid_data_output=training_outputs[401:-1,:]
#print(training_inputs.shape)

#参数定义,设计一层隐藏层神经元，个数为8，输入层神经元个数是6，输出层神经元个数是1，6*8*1   得出两个权值矩阵6*8  和8*1
#每层的激活函数都设置为leak_relu
np.random.seed(1)
weights=[]
weights_one = 2 * np.random.random((6,8))
weights_two = 3 * np.random.random((8,1))
weights.append(weights_one)
weights.append(weights_two)

iterations=5000
alpha=[]
alpha_one=0.05
alpha_two=0.02
alpha.append(alpha_one)
alpha.append(alpha_two)

#print(leakrelu(training_outputs,0.1))
#print(training_inputs)
#print(np.shape(training_inputs))
#print(np.shape(weights))
#print(np.shape(training_outputs))

#开始训练
weights_res=train(train_data_input,train_data_output,weights,iterations,alpha)

#使用训练的模型对验证集进行验证

y_estimates=prediction(prediction(valid_data_input,weights_res[0]),weights_res[1])
error_estimastes=(y_estimates-valid_data_output).sum()
print(error_estimastes)


#使用训练的模型进行预测
input_num=[]
input_nums=[]
input_num.append(recordData[-1][2])
input_num.append(recordData[-1][3])
input_num.append(recordData[-1][4])
input_num.append(recordData[-1][5])
input_num.append(recordData[-1][6])
input_num.append(recordData[-1][7])
input_nums.append(input_num)
input_nums=np.array(input_num)

#normalize_data(recordData[:,2])
#print(type(recordData))

mean_d=training_inputs_d.mean(axis=0)
max_d=training_inputs_d.max(axis=0)
min_d=training_inputs_d.min(axis=0)

#print(max_d)
#print(min_d)
#mean_t=array_oneTotwo(mean_d)
#max_t=array_oneTotwo(max_d)
#min_t=array_oneTotwo(min_d)
#print(array_oneTotwo(mean_d))
#print(mean_d.shape)
#print(np.shape(training_inputs))
#input_nums.reshape(-1)

input_nums_normalize=(input_nums-mean_d)/(max_d-min_d)
#print(input_nums_normalize)
#print(mean_d)
#print(max_d)
#print(min_d)
input_nums_normalize.reshape(1,6)

#result_predic=np.dot(np.dot(input_nums_normalize,weights_res[0]),weights_res[1])
result_predic=prediction(prediction(input_nums_normalize,weights_res[0]),weights_res[1])


#输出预测结果
print(result_predic)

#x_data=np.linspace(0,1,num=y_estimates.shape[1])
y1_data=y_estimates[:,0]
y2_data=valid_data_output[:,0]
error_estimastes_data=(y_estimates-valid_data_output)
y3_data=error_estimastes_data[:,0]

plf.plot(y1_data,'-r')
plf.plot(y2_data,'-b')
plf.plot(y3_data,'-g')
plf.legend(labels=['y_estimastes','y_realvalues','error_values'])

plf.show()

#error_estimastes=(y_estimates-valid_data_output)
print(result_predic*(training_outputs_d.max(axis=0)-training_outputs_d.min(axis=0))+training_outputs_d.mean(axis=0))
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

