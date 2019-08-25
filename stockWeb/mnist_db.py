#coding:utf8

import pandas as pd
import numpy as np
import scipy.optimize as opt
#from scipy.optimize import fmin_slsqp
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plf
import requests
import pymysql


class mnist_data():
	def __init__(self):
		self.hostname='localhost'
		self.username='root'
		self.password='1234'
		self.dbname='test'
		self.port='3367'
		self.charset='utf8'
	def create_db(self):
		conn=pymysql.connect(host=self.hostname,user=self.username,password=self.password,database=self.dbname,charset=self.charset)
		cursor=conn.cursor()
		sql = "create table `mnistflowers`(\
	    `id` int(10) NOT NULL AUTO_INCREMENT ,\
	    `sepallength` double NOT NULL ,\
	    `sepalwidth` double NOT NULL ,\
	    `petallength` double NOT NULL ,\
	    `petalwidth` double NOT NULL ,\
	    `dataclass` char(30) NOT NULL ,\
	    PRIMARY KEY (`id`))CHARSET=utf8"
		cursor.execute(sql)
		conn.commit()
		result = cursor.fetchone()
		cursor.close()
		conn.close()
		return result
	def insert_of_table(self,sepal_length,sepal_width,petal_length,petal_width,data_class):
		conn=pymysql.connect(host=self.hostname,user=self.username,password=self.password,database=self.dbname,charset=self.charset)
		cursor=conn.cursor()

		sql="Insert INTO `mnistflowers` (`sepallength`,`sepalwidth`,`petallength`,`petalwidth`,`dataclass`) VALUES (%s,%s,%s,%s,%s)"

		cursor.execute(sql,(sepal_length,sepal_width,petal_length,petal_width,data_class))
		
		conn.commit()
		result = cursor.fetchone()
		cursor.close()
		conn.close()
		return result
	def query_by_class(self,data_class):
		conn=pymysql.connect(host=self.hostname,user=self.username,password=self.password,database=self.dbname,charset=self.charset)
		cursor=conn.cursor()
		sql="SELECT * FROM `mnistflowers` WHERE `dataclass` = %s"

		cursor.execute(sql,(data_class))
		conn.commit()
		result = cursor.fetchall()
		cursor.close()
		conn.close()
		return result
	def query_for_table(self):
		conn=pymysql.connect(host=self.hostname,user=self.username,password=self.password,database=self.dbname,charset=self.charset)
		cursor=conn.cursor()
		sql="SELECT * FROM `mnistflowers`"

		cursor.execute(sql)
		conn.commit()
		result = cursor.fetchall()
		cursor.close()
		conn.close()
		return result
	def get_data(self):
		r_data=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
		with open('./data/iris.data','w') as f:
			f.write(r_data.text)
		df=pd.read_csv('./data/iris.data',names=['sepal length','sepal width','petal length','petal width','class'])
		return df
	def data_intodb(self,df):
		for a,b,c,d,e in zip(df['sepal length'].values,df['sepal width'].values,df['petal length'].values,df['petal width'].values,df['class'].values):
			#print(float(a),float(b),float(c),float(d),e)
			#self.insert_of_table(a,b,c,d,e)
			self.insert_of_table(float(a),float(b),float(c),float(d),e)
	def db_to_np(self):
		data_all=self.query_for_table()
		data_all_csv=[]
		for each_data in data_all:
			each_temp=[]
			each_temp.append(each_data[1])
			each_temp.append(each_data[2])
			each_temp.append(each_data[3])
			each_temp.append(each_data[4])
			each_temp.append(each_data[5])
			data_all_csv.append(each_temp)
		return np.array(data_all_csv)
		
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
	mnist_one=mnist_data()
	###########create minist db############ 
	#df_one=mnist_one.create_db()
	###########put mnistdata insert into db###########
	#df=mnist_one.get_data()
	#mnist_one.data_intodb(df)
	###########exact data from db###########
	#data_all=mnist_one.query_for_table()
	#data_class=mnist_one.query_by_class('Iris-virginica')
	data_np=mnist_one.db_to_np()
	print(data_np)
	# X_one=np.array(df_one['petal length'].values)
	# X_two=np.empty((X_one.shape[0],1))
	# for i in range(X_one.shape[0]):
		# X_two[i,:]=X_one[i]
	# #print(X_two.shape,X_two)
	# X=np.insert(X_two,0,np.zeros(X_one.shape[0])+1,axis=1)
	
	# #print(X)
	# Y=np.array(df_one['petal width'].values)
	# X_three=np.insert(X_two,1,Y,axis=1)
	# print(X_three)
	# Y_two=np.empty((X_one.shape[0],1))
	# for i in range(X_one.shape[0]):
		# Y_two[i,:]=Y[i]
	# Y_three=np.empty((X_one.shape[0],1))
	# for i in range(X_one.shape[0]):
		# if X_one[i] < 2.5:
			# Y_three[i,:]=-1
		# else:
			# Y_three[i,:]=1
	# print(Y_three)
	# svm_one.linear_fit(X_one,Y,X,Y)
	# svm_one.svm_linear_fit(X_one,Y,X_three,Y_three)
	




