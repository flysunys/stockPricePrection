#coding:utf8

import pandas as pd
import numpy as np
import scipy.optimize as opt
#from scipy.optimize import fmin_slsqp
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plf
import requests
from mnist_db import mnist_data


class svm_linear():
	def __init__(self):
		self.c=0.01
		self.alpha=0.2
	def get_data(self):
		r_data=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
		with open('./data/iris.data','w') as f:
			f.write(r_data.text)
		df=pd.read_csv('./data/iris.data',names=['sepal length','sepal width','petal length','petal width','class'])
		return df 
	def figure_plot(self,data):
		plf.plot(data)
		plf.show()
	def figure_scatter(self,x_data,y_data):
		plf.scatter(x_data,y_data)
		plf.show()
	def linear_fit(self,x_data,y_data,X,Y):
		reg=linear_model.LinearRegression()
		reg.fit(X,Y)
		coef_data=reg.coef_
		x_test=np.linspace(0,20,num=20)
		plf.scatter(x_data,y_data)
		plf.plot(x_test,coef_data[1]*x_test+coef_data[0])
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
	svm_one=svm_linear()
	#df_one=svm_one.get_data()
	df_two=mnist_data.
	print(df_two)
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
	




