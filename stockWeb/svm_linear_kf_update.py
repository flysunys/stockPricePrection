#coding:utf8

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plf


#数据准备
x=[[1,2,3,4,5,7,9,11],[4,2,5,3,6,7,5,7]]
x1=[[1,2,3,4],[4,2,5,3]]
x2=[[5,7,9,11],[6,7,5,7]]
y=[[-1,-1,-1,-1,1,1,1,1]]

#plf.scatter(x1_data,y1_data)
#plf.show()

x_data=np.array(x)
x1_data=np.array(x1)
x2_data=np.array(x2)
y_data=np.array(y)
one_data=np.ones((y_data.shape[0],1))
#print(x1_data[0,:].shape)

#plf.scatter(x1_data[0,:],x1_data[1,:],marker='o')
#plf.scatter(x2_data[0,:],x2_data[1,:],marker='x')
#plf.show()

X=np.insert(x_data.T,0,values=one_data,axis=1)
Y=y_data.T
#print(X)


#参数设置

sigma=0.3

theta=6*np.random.random((X.shape[1],1))

theta[0,:]=1

theta=theta/np.sqrt(np.dot(theta.T,theta)-1)

#print(theta)

#c_para=1.0/X.shape[1]

c_para=1.5

J_cost=6

threshold=0.1

alpha=0.002   #学习率


grand=np.array([[1],[2],[3]])

maxgen = 5000

#使用梯度下降法寻找最小化近似最优解，受参数的影响很大

for i in range(maxgen):

	Y_etsimates=np.dot(X,theta)

	idx=np.where(Y*Y_etsimates<1)[0]

	#print(idx)

	if not any(idx):
		print("idx is empty and clutter is right!")
		break
	else:
		e=Y[idx]-Y_etsimates[idx]
		print(e)
		cost_func=c_para*np.dot(e.T,e)
		grand=-2*np.dot(X[idx].T,e)
		#print(cost_func)
		print(grand)
		theta = (1-alpha)*theta - alpha*grand
		#theta=theta/np.sqrt(np.dot(theta.T,theta)-1)
		print(theta)
		
x_test=[1,2,3,4,5,6,7,8,9,10]
y_test=(-theta[1,:]*np.array(x_test)-theta[0,:])/theta[2,:]
plf.scatter(np.array(x_test),y_test,marker='^')
plf.scatter(x1_data[0,:],x1_data[1,:],marker='o')
plf.scatter(x2_data[0,:],x2_data[1,:],marker='x')
plf.show()
	




