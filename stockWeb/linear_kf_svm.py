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
		if list(self.kValue.keys())[0]=='linear':  #如果是线性核，则使用线性核计算向量的内积
			each_k=np.dot(XData,A.T)   #m*n维的XData和1*n维向量相乘，得到m个向量的内积组成的向量
		elif list(self.kValue.keys())[0]=='gaussian':
			for j in range(m):
				delta=XData[j,:]-A
				each_k[j]=np.dot(delta,delta.T)
			each_k=np.exp(each_k/(-self.kValue['gaussian']**2))
		else:
			print("请输入合适的内核")
			raise NameError('can not identify')
		return each_k
	def chooseJ(self,i,Ei):
		maxK=-1;maxDelta=0;Ej=0
		self.eCache[i]=[1,Ei]
		validEcacheList=np.nonzero(self.eCache[:,0])[0]
		if len(validEcacheList)>1:
			for k in validEcacheList:
				if k==i:continue
				Ek=self.calcEk(k)
				deltaE=np.abs(Ei-Ek)
				if deltaE>maxDelta:
					maxK=k;maxDelta=deltaE;Ej=Ek
			return maxK,Ej
		else:
			j=self.randJ(i)
			Ej=self.calcEk(j)
			return j,Ej
	def randJ(self,i):  #随机选取一个非i的整数（0，m-1）
		j=i
		while i==j:
			j=int(np.random.rand()*self.m)
		return j
	def calcEk(self,k):
		return float(np.dot((self.alpha*self.YData).T,self.K[:,k])+self.b)-float(self.YData[k])
	def adjustAlpha(self,aj,H,L):   #调节拉格朗日乘子的大小
		if aj>H:
			aj=H
		if L>aj:
			aj=L
		return aj
	def train(self):
		step=0
		flag=True
		alphaPairsChanged=0
		while(step<self.maxIter) and (alphaPairsChanged>0) or (flag):
			alphaPairsChanged=0
			if flag:
				for i in range(self.m):
					alphaPairsChanged += self.innerLoop(i)
				step += 1
			else:
				nonBoundls=np.nonzero((self.alpha>0)*(self.alpha<self.C))[0]
				for i in nonBoundls:  #对非支持向量的点进行更新
					alphaPairsChanged += self.innerLoop(i)
				step += 1
			if flag:
				flag=False  #返回支持向量的索引和向量值
			elif alphaPairsChanged==0:
				flag=True
		self.supportVectorIndex=np.nonzero(self.alpha>0)[0]
		self.supportVector=self.XData[self.supportVectorIndex]
		self.supportVectorLabel=self.YData[self.supportVectorIndex]
	def innerLoop(self,i):
		Ei=self.calcEk(i)
		#判别公式,alpha中的取值在0，C之间并且误差大于容忍度
		if ((self.YData[i]*Ei<-self.tol) and (self.alpha[i]<self.C)) or ((self.YData[i]*Ei>self.tol) and (self.alpha[i]>0)):
			j,Ej=self.chooseJ(i,Ei)
			alphaIold=self.alpha[i].copy()
			alphaJold=self.alpha[j].copy()
			if self.YData[i] != self.YData[j]:
				L=max(0,self.alpha[j]-self.alpha[i])
				H=min(self.C,self.C+self.alpha[j]-self.alpha[i])
			else:
				L=max(0,self.alpha[j]+self.alpha[i]-self.C)
				H=min(self.C,self.alpha[j]+self.alpha[i])
			if H==L:
				return 0
			W=self.K[i,i]+self.K[j,j]-2*self.K[i,j]
			if W < 0:
				return 0
			self.alpha[j]=self.alpha[j]+self.YData[j]*(Ei-Ej)/W
			self.alpha[j]=self.adjustAlpha(self.alpha[j],H,L)
			self.eCache[j]=[1,self.calcEk(j)]
			if np.abs(self.alpha[j]-alphaJold) < 0.00001:
				return 0
			self.alpha[i]=self.alpha[i]+self.YData[i]*self.YData[j]*(alphaJold-self.alpha[j])
			self.eCache[i]=[1,self.calcEk(i)]
			bi=-Ei-self.YData[i]*self.K[i,i]*(self.alpha[i]-alphaIold)-self.YData[j]*self.K[i,j]*(self.alpha[j]-alphaJold)+self.b
			bj=-Ej-self.YData[i]*self.K[i,j]*(self.alpha[i]-alphaIold)-self.YData[j]*self.K[j,j]*(self.alpha[j]-alphaJold)+self.b
			if self.alpha[i]>0 and self.alpha[i]<self.C:
				self.b=bi
			elif self.alpha[j]>0 and self.alpha[j]<self.C:
				self.b=bj
			else:
				self.b=(bi+bj)/2.0
			return 1
		else:
			return 0
	def calcWeight(self):
		self.Weight=np.zeros((self.n,1)).T
		for index in self.supportVectorIndex:
			self.Weight += self.alpha[index] * self.YData[index] * self.XData[index,:]
	def split_line(self,x_data,y_data):
		x_test=np.linspace(0,20,num=20)
		plf.scatter(x_data,y_data)
		plf.plot(x_test,(-self.Weight[:,0]*x_test-self.b[0])/self.Weight[:,1])
		plf.show()
		
	def predict(self,test_one_x):
		pre_y=np.dot(test_one_x,self.Weight.T)
		index1=np.where(pre_y>=0)
		index2=np.where(pre_y<0)
		result_pre=np.zeros(np.shape(pre_y))
		result_pre[index1]=1
		result_pre[index2]=-1
		return result_pre
		
				
	
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
	met = svm_linear_kf_data()
	#选取2/3为训练集。1/3为测试集
	train_features, test_features, train_labels, test_labels = train_test_split(
        np_data_two[:,0:2], met.array_oneTotwo(np_data_two[:,2]), test_size=0.3, random_state=23323)
	#print(train_features)
	#train_features = rebuild_features(train_features)
	#test_features = rebuild_features(test_features)
	print('Start training')
	time_2 = time.time()
	#met = svm_linear_kf_data()
	met.C=0.7
	met.tol=0.001
	met.maxIter=100
	met.kValue['linear']=1
	met.initparam(train_features,train_labels)
	met.train()

	time_3 = time.time()
	print('training cost ', time_3 - time_2, ' second', '\n')
	met.calcWeight()
	print('Start predicting')
	
	print(met.Weight)
	print(met.b)
	
	met.split_line(np_data_two[:,0],np_data_two[:,1])
	test_predict = met.predict(test_features)
	time_4 = time.time()
	print('predicting cost ', time_4 - time_3, ' second', '\n')

	score = accuracy_score(met.array_twoToone(test_labels), test_predict)
	print("The accruacy socre is ", score)
	
	




