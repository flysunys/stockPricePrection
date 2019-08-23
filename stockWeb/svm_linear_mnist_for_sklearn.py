#coding:utf8

import pandas as pd
import numpy as np
import scipy.optimize as opt
#from scipy.optimize import fmin_slsqp
from sklearn import svm
import matplotlib.pyplot as plf
import requests


class svm_linear():
	def __init__(self):
		self.c=0.01
		self.alpha=0.2
	def get_data(self):
		r_data=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
		with open('./data/iris.data','w') as f:
			f.write(r_data.text)
		df=pd.read_csv('./data/iris.data',names=['sepal length','sepal width','petal length','petal width','class'])
		return df

if __name__=='__main__':
	svm_one=svm_linear()
	#print(svm_one.c)
	df_one=svm_one.get_data()
	#print(df_one['sepal length'].values)




