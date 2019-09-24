#coding:utf8

#@author:flysun
#@date:2019-09-23
#@last modified by:flysun
#@last modified time:2019-09-23

"""
description:

svm又叫最大间隔分类器，线性可分支持向量机

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
		self.c=0.01
		self.alpha=0.2
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
	def svm_linear_kf(self,X,Y):
		theta=6*np.random.random((X.shape[1]+1,1))
		ones_data=np.zeros(X.shape[0])+1
		X=np.insert(X,0,ones_data,axis=1)
		alpha_params=np.zeros(X.shape[0])+1
		
	
			
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
	
	




