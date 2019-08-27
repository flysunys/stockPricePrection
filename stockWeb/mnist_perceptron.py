#coding:utf8

import pandas as pd
import numpy as np
import scipy.optimize as opt
#from scipy.optimize import fmin_slsqp
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plf
import requests
#from mnist_db import mnist_data  #对于同一目录下调用可以如此
from DButils.mnist_database import mnist_data  #调用子目录下的python文件，需要加目录名称


class perceptron_data():
	def __init__(self):
		self.c=0.01
		self.alpha=0.2
	def figure_plot(self,data):
		plf.plot(data)
		plf.show()
	def figure_scatter(self,x_data,y_data):
		plf.scatter(x_data,y_data)
		plf.show()
	def perceptron_fit_normal(self,x_data,y_data,X,Y,iters,alpha):
		#set paramter
		theta=3*np.random.random((X.shape[1]+1,1))
		ones_data=np.zeros(X.shape[0])+1
		X=np.insert(X,0,ones_data,axis=1)
		#print(type(X[1,1]))
		for i in range(iters):
			Y_estimates=np.dot(X,theta)
			idx=np.where(Y_estimates*Y<0)
			if len(idx)==0:
				print("全部分类正确")
				return theta
			print(idx)
			cost_per=-np.sum(Y_estimates[idx]*Y[idx])
			grand_w=-np.dot(X[idx].T,Y[idx])
			theta=theta+alpha*grand_w
		coef_data=theta
		x_test=np.linspace(0,20,num=20)
		plf.scatter(x_data,y_data)
		plf.plot(x_test,(-coef_data[1,:]*x_test-coef_data[0,:])/coef_data[2,:])
		plf.show()
		
	def svm_linear_fit(self,x_data,y_data,X,Y):
		svm_model= svm.LinearSVC()
		svm_model.fit(X,Y)
		w=svm_model.coef_
		b=svm_model.intercept_
		print(w,b)
		x_test=np.linspace(0,20,num=20)
		plf.scatter(x_data,y_data)
		plf.plot(x_test,(-w[0,0]*x_test-b[0])/w[0,1])
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
	df_data_one_replace=df_data_one.replace('Iris-setosa',-1)
	df_data_two=df_data_one_replace.replace('Iris-versicolor',1)
	########covert data to np.array
	df_data_two = pd.DataFrame(df_data_two, dtype='float')
	np_data_two=df_data_two.values
	np_data_two.astype(float)
	########instance to object 
	instance_one=perceptron_data()
	instance_one.perceptron_fit_normal(np_data_two[:,0],np_data_two[:,1],np_data_two[:,0:2],np_data_two[:,2],1000,0.2)
	instance_one.figure_scatter(np_data_two[:,0],np_data_two[:,1])
	########
	X_one=np.array(df_one['petal length'].values)
	X_two=np.empty((X_one.shape[0],1))
	for i in range(X_one.shape[0]):
		X_two[i,:]=X_one[i]
	X=np.insert(X_two,0,np.zeros(X_one.shape[0])+1,axis=1)
	Y=np.array(df_one['petal width'].values)
	X_three=np.insert(X_two,1,Y,axis=1)
	print(X_three)
	Y_two=np.empty((X_one.shape[0],1))
	for i in range(X_one.shape[0]):
		Y_two[i,:]=Y[i]
	Y_three=np.empty((X_one.shape[0],1))
	for i in range(X_one.shape[0]):
		if X_one[i] < 2.5:
			Y_three[i,:]=-1
		else:
			Y_three[i,:]=1
	print(Y_three)
	svm_one.linear_fit(X_one,Y,X,Y)
	svm_one.svm_linear_fit(X_one,Y,X_three,Y_three)
	




