#coding:utf8

#author:flysun
#date:2019-09-14
#description:决策树的简单思想是通过计算每个特征信息增益值，选取待分类的特征，递归循环，然后计算每类数据的
#其他特征，直到没有可分的特征或者直到数据不可分为止

import pandas as pd
import numpy as np
import scipy.optimize as opt
#from scipy.optimize import fmin_slsqp
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plf
import requests
#from mnist_db import mnist_data  #对于同一目录下调用可以如此
from math import log
import operator
from DButils.mnist_database import mnist_data  #调用子目录下的python文件，需要加目录名称

"""
决策树常用的算法有id3，c4.5和cart算法，每种算法对于决策树的特征选择，决策树生成和剪枝的过程有所描述
ID3算法存在的缺点： 
1. ID3算法在选择根节点和内部节点中的分支属性时，采用信息增益作为评价标准。信息增益的缺点是倾向于选择取值较多是属性，在有些情况下这类属性可能不会提供太多有价值的信息。 
2. ID3算法只能对描述属性为离散型属性的数据集构造决策树
"""


class DecisionTree():
	def __init__(self):
		self.k=3
	def array_oneTotwo(self,x_array):
		t=np.empty((x_array.shape[0],1))
		for i in range(x_array.shape[0]):
			t[i,:]=x_array[i]
		return t
	def calcShannonEnt(self,dataSet):  # 计算数据的熵(entropy)
		numEntries=len(dataSet)  # 数据条数
		labelCounts={}
		for featVec in dataSet:
		currentLabel=featVec[-1] # 每行数据的最后一个字（类别）
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1  # 统计有多少个类以及每个类的数量
		shannonEnt=0
		for key in labelCounts:
			prob=float(labelCounts[key])/numEntries # 计算单个类的熵值
			shannonEnt-=prob*log(prob,2) # 累加每个类的熵值
		return shannonEnt
	def calculate_probility(self,X):
		m,n=np.shape(X)
		for j in range(n):
    
		
	
		
		
		
			
	

if __name__=='__main__':
	#########get data from database
	
	
	
	
	




