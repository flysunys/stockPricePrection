#coding:utf8

#author:flysun
#date:2019-10-01
"""
description

adaboost提升算法：

1、首先构建阈值分类器，针对特征的每一维，寻找它的最优阈值 使得 特征分类的 误差最小

2、对所有维的特征进行阈值分类，得到每一轮分类的 分类误差率，得到 每一轮的分类器和分类系数

3、使用得到的分类系数和分类器，叠加得到 最终分类器


"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metric import accuracy_score
import matplotlib.pyplot as plf
import requests
#from mnist_db import mnist_data  #对于同一目录下调用可以如此
from DButils.mnist_database import mnist_data  #调用子目录下的python文件，需要加目录名称
import time
import math
import logging


sign_time_count=0

class Sign():
	"""
	阈值分类器
	有两种方向，
	1）x<v  y=1
	2）x>v  y=1
	v是阈值轴
	
	v的取值范围是该特征维度下的最大值到最小值之间的所有值
	
	"""
	def __init__(self,features,labels,w):
		self.X = features    #训练数据集的特征
		self.Y = labels      #训练数据集的标签
		self.N = len(labels) #训练数据的数量
		self.w = w           #训练数据的权值分布
		
		f_c = [each_f for each_f in features]
		self.indexes=np.linspace(round(min(f_c)),round(max(f_c)),num=round(max(f_c))-round(min(f_c))+1)
		
	def _train_less_than_(self):
		"""
		寻找（x<v  y=1）情况下的最优v
		"""
		index = -1
		error_score = 1000000
		for i in self.indexes:
			score = 0
			for j in range(self.N):
				val= -1
				if self.X[j]<i:
					val = 1
				if val*self.Y[j]<0:
					score += self.w[j]
			if score < error_score:
				index = i
				error_score = score
		return index,error_score
	
	def _train_more_than_(self):
		"""
		寻找 （x>v y=1）情况下的最优v
		"""
		index=-1
		error_score = 1000000
		for i in self.indexes:
			score = 0
			for j in range(self.N):
				val = 1
				if self.X[j]<i:
					val = -1
				if val*self.Y[j]<0:
					score += self.w[j]
			if score < error_score:
				index =i
				error_score = score
		return index,error_score
	def train(self):
		global sign_time_count
		time1 = time.time()
		less_index,less_score=self._train_less_than_()
		more_index,more_score=self._train_more_than_()
		time2=time.time()
		sign_time_count += time2-time1
		if less_score < more_score:
			self.is_less=True
			self.index=less_index
			return less_score
		else:
			self.is_less=False
			self.index=more_index
			return more_score
	def predict(self,feature):
		if self.is_less>0:
			if feature<self.index:
				return 1.0
			else:
				return -1.0
		else:
			if feature<self.index:
				return 1.0
			else:
				return -1.0
				
					



class adaboost_data():
	def __init__(self):
		pass
	def __init_parameters_(self,features,labels):
		self.X = features   #训练集特征
		self.Y = labels     #训练集标签
		self.n = len(features[0])  #特征维度
		self.N = len(features)  #训练集大小
		self.M = 10
		
		self.w = [1.0/self.N]*self.N   #训练集权值分布
		self.alpha = []                #分类器系数
		self.classifier = []           #（维度，分类器），针对当前维度的分类器
		
	def _w_(self,index,classifier,i):   #更新训练集的权值分布
		return self.w[i]*math.exp(-self.alpha[-1]*self.Y[i]*classifier.predict(self.X[i][index]))
	def _Z_(self,index,classifier):  #计算规范化因子
		Z=0
		for i in range(self.N):
			Z += self._w_(index,classifier,i)
		return Z
		
	def train(self,features,labels):
		self._init_parameters_(features,labels)
		for times in range(self.M):
			logging.debug('iterater %d' % times)
			time1 = time.time()
			map_time = 0
			
			best_classifier = (100000,None,None)   #(误差率，针对的特征，分类器)
			for i in range(self.n):
				map_time -=time.time()
				features = map(lambda x:x[i],self.X)
				map_time +=time.time()
				classifier = Sign(features,self.Y,self.w)
				error_score = classifier.train()
				
				if error_score < best_classifier[0]:
					best_classifier = (error_score,i,classifier)
			em = best_classifier[0]
			
			print("em is %s, index is %d" % (str(em),best_classifier[1]))
			time2=time.time()
			global sign_time_count
			
			print("总运行时间:%s, 那两段关键代码运行时间:%s, map的时间是:%s" % (str(time2-time1),str(sign_time_count),str(map_time)))
			sign_time_count = 0
			if em == 0:
				self.alpha.append(100)
			else:
				self.alpha.append(0.5*math.log((1-em)/em))
			self.classifier.append(best_classifier[1:])
			Z = self._Z_(best_classifier[1],best_classifier[2])
			
			#计算训练集权值分布
			
			for i in range(self.N):
				self.w[i] = self._w_(best_classifier[1],best_classifier[2],i)/Z
	def _predict_(self,feature):
		result = 0.0
		for i in range(self.M):
			index = self.classifier[i][0]
			classifier = self.classifier[i][1]
			result += self.alpha[i]*classifier.predict(feature[index])
		if result>0:
			return 1
		return -1
	def predict(self,features):
		results = []
		for feature in features:
			results.append(self._predict_(feature))
		return results
	
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
	df_data_two = pd.DataFrame(df_data_two)
	df_data_two=df_data_two.astype('float')
	np_data_two=df_data_two.values
	#np_data_two=np_data_two.astype('float')
	########instance to object 
	#instance_one=perceptron_data()
	########set paramters
	iters=500
	alpha=0.1
	learningrate=1
	
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	train_features, test_features, train_labels, test_labels = train_test_split(
        np_data_two[:,0:2], met.array_oneTotwo(np_data_two[:,2]), test_size=0.3, random_state=23323)
	train_labels = map(lambda x:2*x-1,train_labels)
	ada = adaboost_data()
	ada.train(train_features, train_labels)
	test_predict = ada.predict(test_features)
	test_labels = map(lambda x:2*x-1,test_labels)
    score = accuracy_score(test_labels,test_predict)
	#instance_one.perceptron_fit_pair(np_data_two[:,0],np_data_two[:,1],np_data_two[:,0:2],instance_one.array_oneTotwo(np_data_two[:,2]),iters,learningrate)
	#instance_one.figure_scatter(np_data_two[:,0],np_data_two[:,1])
	########
	
	




