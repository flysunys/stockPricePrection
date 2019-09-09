#coding:utf8

#author:flysun
#date:2019-09-01
#description:knn的简单思想是在k类数据（已经分好的数据）中对于新的一个数据，计算这个数据与其他数据的距离，找出据此数据最近的k个点，然后使用投票的方式决定，该点属于哪一类

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

"""
决策树常用的算法有id3，c4.5和cart算法，每种算法对于决策树的特征选择，决策树生成和剪枝的过程有所描述
"""


class DecisionTree():
	def __init__(self):
		self.k=3
	def array_oneTotwo(self,x_array):
		t=np.empty((x_array.shape[0],1))
		for i in range(x_array.shape[0]):
			t[i,:]=x_array[i]
		return t
	
	def calculate_probility(self,X):
		m,n=np.shape(X)
		for j in range(n):
    			X_unique=np.unique(X[:,j])
   	 		X_probility=[]
    			for i in X_unique:
        			idx=np.where(X[:,j]==i)
        			num_idx=np.shape(idx)[1]
        			print(num_idx)
        			X_probility.append(1.0*num_idx/m)
    			print(X_probility)
    			print(X_unique)
	def calculate_probility_Y(self,Y):
		Y_unique=np.unique(Y)
		Y_probility=[]
		for i in Y_unique:
			#print(np.where(b==i))
			num=np.shape(np.where(Y==i))[1]
			Y_probility.append(1.0*num/np.shape(Y)[0])
		return [Y_unique,Y_probility]
	def calculate_probility_X(self,Y):
		Y_unique=np.unique(Y)
		Y_probility=[]
		for i in Y_unique:
			#print(np.where(b==i))
			num=np.shape(np.where(Y==i))[1]
			Y_probility.append(1.0*num/np.shape(Y)[1])
		return [Y_unique,Y_probility]
		
	def decisiontree_id3(self,X,Y,threshold):
		#set paramter
		m,n=np.shape(X)
		Y_unique=np.unique(Y)
		Y_probility=[]
		for i in Y_unique:
    			#print(np.where(b==i))
    			num=np.shape(np.where(Y==i))[1]
    			#print(num)
    			Y_probility.append(1.0*num/np.shape(Y)[0])
		#print(Y_probility)
		x_data_probility=[]
		#print(np.shape(Y_unique)[0])
		for j in range(np.shape(Y_unique)[0]):
			idx=np.where(Y==Y_unique[j])
			each_Y_probility=[]
			temp_x_data_probility=Y_probility[j]
			for k in range(n):
				#x_k_unique=np.unique(X[idx,k])
				#print(self.calculate_probility_X(X[idx,k]))
				each_Y_probility.append(self.calculate_probility_X(X[idx,k]))
				x_k_probility=self.calculate_probility_X(X[idx,k])
				idx_k=np.where(x_k_probility[0]==x_data[k])
				#print(idx_k)
				if idx_k[0].size:
					temp_x_data_probility*=x_k_probility[1][idx_k[0][0]]
				else:
					temp_x_data_probility*=0
			#print(each_Y_probility)
			#print(temp_x_data_probility)
			x_data_probility.append(temp_x_data_probility)
		#print(x_data_probility)
		return Y_unique[x_data_probility.index(max(x_data_probility))]
		
		
		
			
	

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
	#df_setosa=df_setosa.drop([0,1],axis=1)
	#df_versicolor=df_versicolor.drop([0,1],axis=1)
	#df_virginica=df_virginica.drop([0,1],axis=1)
	########合并df
	df_data_one=pd.concat([df_setosa,df_versicolor,df_virginica], axis=0)
	
	df_data_one_replace=df_data_one.replace('Iris-setosa',0)
	df_data_two=df_data_one_replace.replace('Iris-versicolor',1)
	df_data_three=df_data_two.replace('Iris-virginica',2)
	########covert data to np.array
	df_data_three = pd.DataFrame(df_data_three)
	df_data_three=df_data_three.astype('float')
	#print(pd.cut(df_data_three[0],3,labels=range(3)))
	#print(pd.cut(df_data_three[0],3,labels=[5,6,7]))
	#####discrete data
	df_data_three_dis=df_data_three
	df_data_three_dis[0]=pd.cut(df_data_three[0],3,labels=[5,6,7])
	df_data_three_dis[1]=pd.cut(df_data_three[1],3,labels=[2,3,4])
	df_data_three_dis[2]=pd.cut(df_data_three[2],6,labels=[1,2,3,4,5,6])
	df_data_three_dis[3]=pd.cut(df_data_three[3],4,labels=[0,1,2,3])
	#print(df_data_three_dis)
	np_data_three=df_data_three_dis.values
	#print(np_data_three)
	#np_data_two=np_data_two.astype('float')
	########instance to object
	
	############add test data
	test_data=np.array([5,3,4,2])
	instance_one=NaiveBayes()
	most_probility_test_data=instance_one.Naive_Bayes_Calculate(np_data_three[:,0:4],np_data_three[:,4],test_data)
	########set paramters
	#result
	print(most_probility_test_data)
	
	
	




