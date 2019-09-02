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



class Node():
	def __init__(self,data,lchild=None,rchild=None):
		self.data=data
		self.lchild=lchild
		self.rchild=rchild
	def create_tree(self,dataset,depth):
		if (len(dataset)>0):
			m,n=np.shape(dataset)
			median_index=m//2
			axis_data=depth%n
			sort_dataset=dataset(np.argsort(dataset[:,axis_data]))
			node = Node(sort_dataset[median_index])
			




class knn_data():
	def __init__(self):
		self.k=3
		self.p=2
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
	def knn_normal(self,X,Y,test):
		"""
		普通的knn算法，直接计算预测点和所有点的欧式距离；
		按距离排序，选出距离最小的k个值对应的类；
		使用投票的方式，从k个值选出出现次数最多的类；
		该类就是预测点的所属类
		"""
		#set paramter
		test_r=np.square(X-test)
		#p=2,so this distance is ER diatance
		distance_test=np.sum(test_r,axis=1)
		index_test=np.argsort(distance_test)#从小到大排序的
		#print(index_test)
		#print(index_test[0:7])
		#print(Y[index_test[0:7],0])
		test_k_np=Y[index_test[0:7],0].tolist()
		list_count=[]
		for i in range(self.k):
			list_count.append(test_k_np.count(i))
		return list_count.index(max(list_count))
		#test_0_count=test_k_np.count(0)
		#test_1_count=test_k_np.count(1)
		#test_2_count=test_k_np.count(2)
		#print(test_0_count,test_1_count,test_2_count)
			
	def knn_kd_tree_balance(self,x_data,y_data,X,Y,iters,learnning_rate):
		#set paramter
		#alpha_r=0*np.random.random((X.shape[0],1)) #初始设置为0也可以
		alpha_r=3.5*np.abs(np.random.random((X.shape[0],1)))
		b=0
		#ones_data=np.zeros(X.shape[0])+1
		#X=np.insert(X,0,ones_data,axis=1)  #这里w和b是分开的，对初始数据不需要添加1
		gram_marix=np.dot(X,X.T)
		count_iter=0
		for i in range(iters):
			f_data=np.sum(X*alpha_r*Y,axis=0)
			f_data_two=self.array_oneTotwo(f_data)
			Y_estimates=np.dot(X,f_data_two)+b
			#print(Y_estimates*Y)
			idx=np.where(Y_estimates*Y<=0)
			if len(idx[0])==0:
				print("全部分类正确")
				break
			count_iter+=1
			cost_per=-np.sum(Y_estimates[idx[0],:]*Y[idx[0],:])
			alpha_r[idx[0],:]=alpha_r[idx[0],:]+learnning_rate
			b=b+learnning_rate*np.sum(Y[idx[0],:],axis=0)
		print("迭代总次数是：%s" % (count_iter))
		w_data=f_data
		b_data=b
		x_test=np.linspace(0,20,num=20)
		plf.scatter(x_data,y_data)
		plf.plot(x_test,(-w_data[0]*x_test-b_data)/w_data[1])
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
	np_data_three=df_data_three.values
	#np_data_two=np_data_two.astype('float')
	########instance to object 
	instance_one=knn_data()
	########set paramters
	test_data=np.array([[3,4,5,1]])
	#使用测试数据测试普通的knn算法
	test_kind=instance_one.knn_normal(np_data_three[:,0:4],instance_one.array_oneTotwo(np_data_three[:,4]),test_data)
	print(test_kind)
	#instance_one.figure_scatter(np_data_two[:,0],np_data_two[:,1])
	########
	
	




