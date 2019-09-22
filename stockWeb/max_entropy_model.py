#coding:utf8

#@author:flysun
#@date:2019-09-20
#@last modified by:flysun
#@last modified time:2019-09-22

"""
description:

最大熵原理是用于选择一个最优的概率模型，熵最大的模型就是最优概率模型，属于模型优化的范围

将最大熵原理应用到分类得到了最大熵模型，也就是最大熵模型是一个分类模型

对于多分类的情况，如果各类是明显分开的，且各类中心不在一条直线上，使用分别训练各类的线性权值的方式是可行的；

如果各类的中心在一条直线上，直接使用线性分类模型就不可行了

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



class max_entropy_model_data():
	def init_params(self,X,Y):
		self.learningrate=0.2
		self.X_=X
		self.Y_=set()
		self.cal_Pxy_Px(X,Y)
		self.N=len(X)  #训练集大小
		self.n=len(self.Pxy)    #
		self.M=10000.0
		self.build_dict()
		self.cal_EPxy()
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
	def cal_Pxy_Px(self,X,Y):
		self.Pxy=defaultdict(int)
		self.Px=defaultdict(int)
		for i in range(len(X)):
			x_,y_array = X[i],Y[i]
			y=y_array[0]
			self.Y_.add(y)
			for x in x_:
				self.Pxy[(x,y)] +=1
				self.Px[x] +=1
	def cal_EPxy(self):
		"""
		特征函数关于经验分布的期望值
		"""
		self.EPxy=defaultdict(int)
		for id in range(self.n):
			(x,y)=self.id2xy[id]
			self.EPxy[id]=float(self.Pxy[(x,y)])/float(self.N)
			
	def build_dict(self):
		self.id2xy={}
		self.xy2id={}
		for i,(x,y) in enumerate(self.Pxy):
			self.id2xy[i]=(x,y)
			self.xy2id[(x,y)]=i
			
	def cal_pxy(self,X,y):
		result=0.0
		for x in X:
			if self.fxy(x,y):
				id = self.xy2id[(x,y)]
				print(id)
				result += self.w[id]
		return (math.exp(result),y)
		
	def cal_probality(self,X):
		"""
		计算最大熵模型对应的条件概率分布
		"""
		Pyxs=[(self.cal_pxy(X,y)) for y in self.Y_]
		Z=sum([prob for prob,y in Pyxs])
		return [(prob/Z,y) for prob,y in Pyxs]
		
	def cal_EPx(self):
		"""
		计算特征函数关于模型pyx和模型px的期望值
		"""
		self.EPx=[0.0 for i in range(self.n)]
		
		for i,X in enumerate(self.X_):
			Pyxs = self.cal_probality(X)
			
			for x in X:
				for Pyx,y in Pyxs:
					if self.fxy(x,y):
						id=self.xy2id[(x,y)]
						self.EPx[id] += Pyx*(1.0/self.N)
						
	def fxy(self,x,y):
		return (x,y) in self.xy2id
		
	def train(self,X,Y):
		self.init_params(X,Y)
		self.w=[0.0 for i in range(self.n)]
		
		max_iterations = 1000
		
		for times in range(max_iterations):
			print('iterates times %d' % times)
			sigmas=[]
			self.cal_EPx()
			
			for i in range(self.n):
				sigma = 1.0 / self.M * math.log(self.EPxy[i]/self.EPx[i])
				sigmas.append(sigma)
			self.w=[self.w[i]+sigmas[i] for i in range(self.n)]
			
	def predict(self,testset):
		results=[]
		for test in testset:
			result=self.cal_probality(test)
			results.append(max(result,key=lambda x:x[0])[1])
		return results
			
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
	########set paramters
	iters=50000
	alpha=0.08
	learningrate=0.00005
	#如果theta初始值的斜率是正，收敛出的也是正，如果是负，收敛出的也是负，因为这个数据的可分超平面太多了，明显的线性可分
	#logistic 2 分类
	logger=logging.getLogger()
	logger.setLevel(logging.INFO)
	#选取2/3为训练集。1/3为测试集
	train_features, test_features, train_labels, test_labels = train_test_split(
        np_X_three[:,0:2], Y_Three, test_size=0.33, random_state=23323)
	#print(train_features)
	train_features = rebuild_features(train_features)
	test_features = rebuild_features(test_features)
	print('Start training')
	time_2 = time.time()
	met = max_entropy_model_data()
	met.train(train_features, train_labels)

	time_3 = time.time()
	print('training cost ', time_3 - time_2, ' second', '\n')

	print('Start predicting')
	test_predict = met.predict(test_features)
	time_4 = time.time()
	print('predicting cost ', time_4 - time_3, ' second', '\n')

	score = accuracy_score(met.array_twoToone(test_labels), test_predict)
	print("The accruacy socre is ", score)
	
	




