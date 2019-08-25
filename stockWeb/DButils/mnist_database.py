#coding:utf8

import pandas as pd
import numpy as np
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



