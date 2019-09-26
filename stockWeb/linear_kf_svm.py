#coding:utf8

#@author:flysun
#@date:2019-09-23
#@last modified by:flysun
#@last modified time:2019-09-23

"""
description:

svm又叫最大间隔分类器，线性可分支持向量机

使用smo算法解决线性可分数据

"""

import pandas as pd
import time
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.optimize as opt
#from scipy.optimize import fmin_slsqp
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plf
import requests
import logging
from collections import defaultdict 
#from mnist_db import mnist_data  #对于同一目录下调用可以如此
from DButils.mnist_database import mnist_data  #调用子目录下的python文件，需要加目录名称



class svm_linear_kf_data():
	def __init__(self):
		self.C=0 #惩罚因子
		self.tol=0 #容错率
		self.b=0 #截距
		self.kValue={} #设置核函数是线性的还是高斯的
		self.maxIter=1000  #最大迭代次数
		self.supportVectorIndex=[]  #支持向量的下标
		self.supportVector=[]  #支持向量	
	def array_oneTotwo(self,x_array):
		t=np.empty((x_array.shape[0],1))
		for i in range(x_array.shape[0]):
			t[i,:]=x_array[i]
		return t
	def array_twoToone(self,x_array):
		t=np.zeros(x_array.shape[0])
		for i in range(x_array.shape[0]):
			t[i]=x_array[i][0]
		return t
	def figure_plot(self,data):
		plf.plot(data)
		plf.show()
	def figure_scatter(self,x_data,y_data):
		plf.scatter(x_data,y_data)
		plf.show()
	def initparam(self,X,Y):
		self.XData=X
		self.YData=Y
		m,n=np.shape(X)
		self.m=m
		self.n=n
		self.alpha=np.zeros((self.m,1))
		self.eCache=np.zeros((self.m,2))
		self.K=np.zeros((self.m,self.m))
		for i in range(self.m):
			self.K[:,i]=self.kernels(self.XData,self.XData[i,:])
	def kernels(self,XData,A):   #根据核的特性，计算核函数的第i列的所有值
		m,n=np.shape(XData)
		each_k=np.zeros((m,1))
		if self.kValue.keys()[0]=='linear':  #如果是线性核，则使用线性核计算向量的内积
			each_k=np.dot(XData,A.T)   #m*n维的XData和1*n维向量相乘，得到m个向量的内积组成的向量
		elif self.kValue.keys()[0]=='gaussian':
			for j in range(m):
				delta=XData[j,:]-A
				each_k[j]=np.dot(delta,delta.T)
			each_k=np.exp(each_k/(-self.kValue['gaussian']**2))
		else:
			print("请输入合适的内核")
			raise NameError('can not identify')
		return each_k
			
		
		
	
			
def rebuild_features(features):
	"""
	将原feature的（a0,a1,a2,a3,a4,...）
    变成 (0_a0,1_a1,2_a2,3_a3,4_a4,...)形式
	"""
	new_features=[]
	for feature in features:
		new_feature=[]
		for i,f in enumerate(feature):
			new_feature.append(str(i)+'_'+str(f))
		new_features.append(new_feature)
	return new_features
	

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
	df_data_two = pd.DataFrame(df_data_two)
	df_data_two=df_data_two.astype('float')
	np_data_two=df_data_two.values
	########set paramters
	iters=50000
	alpha=0.08
	learningrate=0.00005
	#如果theta初始值的斜率是正，收敛出的也是正，如果是负，收敛出的也是负，因为这个数据的可分超平面太多了，明显的线性可分
	#线性可分支持向量机 2 分类
	logger=logging.getLogger()
	logger.setLevel(logging.INFO)
	#选取2/3为训练集。1/3为测试集
	train_features, test_features, train_labels, test_labels = train_test_split(
        np_data_two[:,0:2], instance_one.array_oneTotwo(np_data_two[:,2]), test_size=0.33, random_state=23323)
	#print(train_features)
	#train_features = rebuild_features(train_features)
	#test_features = rebuild_features(test_features)
	print('Start training')
	time_2 = time.time()
	met = svm_linear_kf_data()
	met.train(train_features, train_labels)

	time_3 = time.time()
	print('training cost ', time_3 - time_2, ' second', '\n')

	print('Start predicting')
	test_predict = met.predict(test_features)
	time_4 = time.time()
	print('predicting cost ', time_4 - time_3, ' second', '\n')

	score = accuracy_score(met.array_twoToone(test_labels), test_predict)
	print("The accruacy socre is ", score)
	
	




