#coding:utf8

#author:flysun
#date:2019-09-18
"""
description:

逻辑斯蒂回归模型用于多分类情况，以二分类为例，逻辑斯蒂回归模型的损失函数

"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
#from scipy.optimize import fmin_slsqp
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plf
import requests
import logging
#from mnist_db import mnist_data  #对于同一目录下调用可以如此
from DButils.mnist_database import mnist_data  #调用子目录下的python文件，需要加目录名称



class logistic_data():
	def __init__(self):
		self.c=0.01
		self.alpha=0.2
	def array_oneTotwo(self,x_array):
		t=np.empty((x_array.shape[0],1))
		for i in range(x_array.shape[0]):
			t[i,:]=x_array[i]
		return t
	def figure_plot(self,data):
		plf.plot(data)
		plf.show()
	def figure_scatter(self,x_data,y_data):
		plf.scatter(x_data,y_data)
		plf.show()
	def sigmod(self,x):
		return 1.0/(1.0+np.exp(-x))
	def logistic_fit_normal(self,x_data,y_data,X,Y,iters,alpha):
		#set paramter
		theta=6*np.random.random((X.shape[1]+1,1))
		ones_data=np.zeros(X.shape[0])+1
		X=np.insert(X,0,ones_data,axis=1)
		#print(X)
		#print(type(X[1,1]))
		#print(X.shape,Y.shape)
		count_iter=0
		for i in range(iters):
			Y_estimates=np.dot(X,theta)
			#print(Y_estimates*Y)
			Y_S=self.sigmod(Y_estimates)
			diff=Y_S-Y
			tempw=theta-alpha*np.dot(X.T,diff)
			print(tempw)
			print(np.sqrt(sum((tempw-theta)**2)))
			if np.sqrt(sum((tempw-theta)**2))<0.002:
				print("全部分类正确")
				break
			else:
				theta=tempw
			count_iter+=1
		print("迭代总次数是：%s" % (count_iter))
		coef_data=theta
		x_test=np.linspace(0,20,num=20)
		plf.scatter(x_data,y_data)
		plf.plot(x_test,(-coef_data[1,:]*x_test-coef_data[0,:])/coef_data[2,:])
		plf.show()
		
	def perceptron_fit_pair(self,x_data,y_data,X,Y,iters,learnning_rate):
		#set paramter
		#alpha_r=0*np.random.random((X.shape[0],1)) #初始设置为0也可以
		alpha_r=3.5*np.abs(np.random.random((X.shape[0],1)))
		b=0
		#ones_data=np.zeros(X.shape[0])+1
		#X=np.insert(X,0,ones_data,axis=1)  #这里w和b是分开的，对初始数据不需要添加1
		gram_marix=np.dot(X,X.T)
		count_iter=0
		for i in range(iters):
			f_data=np.sum(X*alpha_r*Y,axis=0)
			f_data_two=self.array_oneTotwo(f_data)
			Y_estimates=np.dot(X,f_data_two)+b
			#print(Y_estimates*Y)
			idx=np.where(Y_estimates*Y<=0)
			if len(idx[0])==0:
				print("全部分类正确")
				break
			count_iter+=1
			cost_per=-np.sum(Y_estimates[idx[0],:]*Y[idx[0],:])
			alpha_r[idx[0],:]=alpha_r[idx[0],:]+learnning_rate
			b=b+learnning_rate*np.sum(Y[idx[0],:],axis=0)
		print("迭代总次数是：%s" % (count_iter))
		w_data=f_data
		b_data=b
		x_test=np.linspace(0,20,num=20)
		plf.scatter(x_data,y_data)
		plf.plot(x_test,(-w_data[0]*x_test-b_data)/w_data[1])
		plf.show()
	
	

if __name__=='__main__':
	#########get data from database
	mnist_one=mnist_data()
	np_setosa=mnist_one.db_to_np_class('Iris-setosa')
	np_versicolor=mnist_one.db_to_np_class('Iris-versicolor')
	#########covert data to pandas dataframe
	df_setosa=pd.DataFrame(np_setosa)
	df_versicolor=pd.DataFrame(np_versicolor)
	df_setosa=df_setosa.drop([0,1],axis=1)
	df_versicolor=df_versicolor.drop([0,1],axis=1)
	########合并df
	df_data_one=pd.concat([df_setosa, df_versicolor], axis=0)
	df_data_one_replace=df_data_one.replace('Iris-setosa',0)
	df_data_two=df_data_one_replace.replace('Iris-versicolor',1)
	########covert data to np.array
	df_data_two = pd.DataFrame(df_data_two)
	df_data_two=df_data_two.astype('float')
	np_data_two=df_data_two.values
	#np_data_two=np_data_two.astype('float')
	########instance to object 
	instance_one=logistic_data()
	########set paramters
	iters=50
	alpha=0.08
	learningrate=1
	#如果theta初始值的斜率是正，收敛出的也是正，如果是负，收敛出的也是负，因为这个数据的可分超平面太多了，明显的线性可分
	#logistic 2 分类
	logger=logging.getLogger()
	logger.setLevel(logging.INFO)
	instance_one.logistic_fit_normal(np_data_two[:,0],np_data_two[:,1],np_data_two[:,0:2],instance_one.array_oneTotwo(np_data_two[:,2]),iters,alpha)
	#logistic 多 分类
	#instance_one.perceptron_fit_pair(np_data_two[:,0],np_data_two[:,1],np_data_two[:,0:2],instance_one.array_oneTotwo(np_data_two[:,2]),iters,learningrate)
	#instance_one.figure_scatter(np_data_two[:,0],np_data_two[:,1])
	########
	
	




