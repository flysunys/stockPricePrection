#coding:utf8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plf

def cost_function_zero(x):
	return -x*(x<1)+(x>1 or x==1)*0
	
def cost_function_one(x):
	return x*(x>-1)+(x<-1 or x==-1)*0

	
def distance_data(x1,x2):
	return np.sum(np.square(x1-x2))


def similarity_data(x1,x2,sigma):
	return np.exp(-distance_data(x1,x2)/(2*sigma*sigma))
	
x1=[1,2,3,-1,-2,5,6]
y1=[2,3,4,-3,-6,2,1]

x1_data=np.array(x1)
y1_data=np.array(y1)

l_data=x1_data

#plf.scatter(x1_data,y1_data)
#plf.show()

#参数设置

sigma=0.3

theta=np.empty((x1_data.size+1))+1.3

theta[0]=1

c_para=1.0/x1_data.size

J_cost=6

threshold=0.3

alpha=0.5   #学习率

#开始计算costfunction and 使用梯度下降收敛参数

while J_cost > threshold:
	J_cost=0
	theta_temp=np.empty((x1_data.size+1))
	theta_temp[0]=1
	for i in range(x1_data.size):
		f=np.empty((x1_data.size+1))
		f[0]=1
		for j in range(l_data.size):
			f[j+1]=similarity_data(x1_data[i],l_data[j],sigma)
		#print(f)
		each_data_cal=y1_data[i]*cost_function_one(np.sum(theta*f))+cost_function_zero(np.sum(theta*f))*(1-y1_data[i])
		#print(each_data_cal)
		theta_temp[i+1]=(alpha*c_para*f[i]*((np.sum(theta*f)<1)*y1_data[i]+(np.sum(theta*f)>-1)*(1-y1_data[i]))+theta[i+1])
		J_cost+=each_data_cal
	J_cost=c_para*J_cost+np.sum(np.square(theta[1:]))/2.0
	theta=theta-theta_temp
	print(J_cost)
	print(theta)
	
print(theta)
	#theta-=(alpha*c_para*x1_data[i]*((theta*f<1)*y1_data[i]+(theta*f>-1)*(1-y1_data[i]))+theta)
	




