#coding:utf8

#author:flysun
#date:2019-09-20
"""
description:

逻辑斯蒂回归模型用于多分类情况，以二分类为例，逻辑斯蒂回归模型的损失函数

对于多分类的情况，如果各类是明显分开的，且各类中心不在一条直线上，使用分别训练各类的线性权值的方式是可行的；

如果各类的中心在一条直线上，直接使用线性分类模型就不可行了

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
		
	def logistic_fit_for_three(self,x_data,y_data,X,Y,iters,learnning_rate):
		#set paramter
		#alpha_r=0*np.random.random((X.shape[0],1)) #初始设置为0也可以
		theta_all=[]
		#theta=6*np.random.random((X.shape[1]+1,1))
		#print(theta.shape)
		#print(X.shape)
		ones_data=np.zeros(X.shape[0])+1
		X=np.insert(X,0,ones_data,axis=1)
		#print(X.shape)
		#print(type(X[1,1]))
		#print(X.shape,Y.shape)
		for j in range(3):
			count_iter=0
			theta=6*np.random.random((X.shape[1],1))
			for i in range(iters):
				Y_estimates=np.dot(X,theta)
				#print(Y_estimates*Y)
				Y_S=self.sigmod(Y_estimates)
				#print(Y_S.shape)
				#print(Y[:,j].shape)
				diff=Y_S-self.array_oneTotwo(Y[:,j])
				#print(diff.shape)
				tempw=theta-learnning_rate*np.dot(X.T,diff)
				#print(tempw.shape)
				#print(np.sqrt(sum((tempw-theta)**2)))
				theta=tempw
				count_iter+=1
			print("迭代总次数是：%s" % (count_iter))
			theta_all.append(theta)
		for j in range(3):
			coef_data=theta_all[j]
			#print(coef_data)
			x_test=np.linspace(0,20,num=20)
			plf.plot(x_test,(-coef_data[1,:]*x_test-coef_data[0,:])/coef_data[2,:])
		plf.scatter(x_data,y_data)
		plf.show()
	
	def logistic_fit_for_three_nolinear(self,x_data,y_data,X,Y,iters,learnning_rate):
		#set paramter
		theta_all=[]
		ones_data=np.zeros(X.shape[0])+1
		X=np.insert(X,0,ones_data,axis=1)
		X=np.insert(X,2,np.square(X[:,1]),axis=1)
		X=np.insert(X,3,X[:,1]**3,axis=1)
		#X=np.insert(X,1,1.0/X[:,1],axis=1)
		#X[:,3]=np.square(X[:,3])
		#print(X.shape)
		#print(type(X[1,1]))
		#print(X.shape,Y.shape)
		for j in range(3):
			count_iter=0
			theta=20*np.random.random((X.shape[1],1))
			for i in range(iters):
				Y_estimates=np.dot(X,theta)
				#print(Y_estimates*Y)
				Y_S=self.sigmod(Y_estimates)
				#print(Y_S.shape)
				#print(Y[:,j].shape)
				diff=Y_S-self.array_oneTotwo(Y[:,j])
				#print(diff.shape)
				tempw=theta-learnning_rate*np.dot(X.T,diff)
				#print(tempw.shape)
				#print(np.sqrt(sum((tempw-theta)**2)))
				theta=tempw
				count_iter+=1
			print("迭代总次数是：%s" % (count_iter))
			theta_all.append(theta)
		for j in range(3):
			coef_data=theta_all[j]
			#print(coef_data)
			x_test=np.linspace(0,20,num=20)
			#plf.plot(x_test,(-coef_data[1,:]*x_test-coef_data[2,:]*(x_test**2)-coef_data[0,:])/coef_data[3,:])
			plf.plot(x_test,(-coef_data[1,:]*x_test-coef_data[2,:]*(x_test**2)-coef_data[3,:]*(x_test**3)-coef_data[0,:])/coef_data[4,:])
			#plf.plot(x_test,(-coef_data[1,:]*(1.0/x_test)-coef_data[2,:]*x_test-coef_data[3,:]*(x_test**2)-coef_data[0,:])/coef_data[4,:])
		plf.scatter(x_data,y_data)
		plf.show()
	
	

if __name__=='__main__':
	#########get data from database
	mnist_one=mnist_data()
	np_setosa=mnist_one.db_to_np_class('Iris-setosa')
	np_versicolor=mnist_one.db_to_np_class('Iris-versicolor')
	np_virginica=mnist_one.db_to_np_class('Iris-virginica')
	#########covert data to pandas dataframe
	df_setosa=pd.DataFrame(np_setosa)
	df_versicolor=pd.DataFrame(np_versicolor)
	df_virginica=pd.DataFrame(np_virginica)
	df_setosa=df_setosa.drop([0,1],axis=1)
	df_versicolor=df_versicolor.drop([0,1],axis=1)
	df_virginica=df_virginica.drop([0,1],axis=1)
	########合并df
	df_data_one=pd.concat([df_setosa, df_versicolor], axis=0)
	df_data_one_replace=df_data_one.replace('Iris-setosa',0)
	df_data_two=df_data_one_replace.replace('Iris-versicolor',1)
	df_data_three=pd.concat([df_setosa,df_versicolor,df_virginica], axis=0)
	#print(df_data_three)
	#print(np.where(df_data_three.loc[:,4]=='Iris-setosa'))
	idx0=np.where(df_data_three.loc[:,4]=='Iris-setosa')
	idx1=np.where(df_data_three.loc[:,4]=='Iris-versicolor')
	idx2=np.where(df_data_three.loc[:,4]=='Iris-virginica')
	#print(df_data_three.shape)
	Y_Three=np.zeros([df_data_three.shape[0],3])
	Y_Three[idx0[0],0]=1
	Y_Three[idx1[0],1]=1
	Y_Three[idx2[0],2]=1
	#print(Y_Three)
	########covert data to np.array
	X_three=df_data_three.drop([4],axis=1)
	#print(X_three)
	df_X_three = pd.DataFrame(X_three)
	df_X_three=df_X_three.astype('float')
	np_X_three=df_X_three.values
	df_data_two = pd.DataFrame(df_data_two)
	df_data_two=df_data_two.astype('float')
	np_data_two=df_data_two.values
	#np_data_two=np_data_two.astype('float')
	########instance to object 
	instance_one=logistic_data()
	########set paramters
	iters=5000
	alpha=0.08
	learningrate=1
	#如果theta初始值的斜率是正，收敛出的也是正，如果是负，收敛出的也是负，因为这个数据的可分超平面太多了，明显的线性可分
	#logistic 2 分类
	logger=logging.getLogger()
	logger.setLevel(logging.INFO)
	#instance_one.logistic_fit_normal(np_data_two[:,0],np_data_two[:,1],np_data_two[:,0:2],instance_one.array_oneTotwo(np_data_two[:,2]),iters,alpha)
	#logistic 多 分类
	#instance_one.logistic_fit_for_three(np_X_three[:,0],np_X_three[:,1],np_X_three[:,0:2],Y_Three,iters,learningrate)
	#logistaic 非线性 多类
	instance_one.logistic_fit_for_three_nolinear(np_X_three[:,0],np_X_three[:,1],np_X_three[:,0:2],Y_Three,iters,learningrate)
	########
	
	




