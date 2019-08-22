#coding:utf8

#这是线性可分的支持向量机，线性可分支持向量机对于线性可分的数据分类效果好

import pandas as pd
import numpy as np
#import scipy.optimize as opt
#from scipy.optimize import fmin_slsqp
from sklearn import svm
import matplotlib.pyplot as plf


def cost_func_kf(theta,*args):
	Y_etsimates=np.dot(X,theta)
	idx=np.where(Y*Y_etsimates<1)[0]
	#print(type(c_para))
	e=Y[idx]-Y_etsimates[idx]
	cost_func=c_para*np.dot(e.T,e)
	grand=-2*np.dot(X[idx].T,e)
	print(cost_func,grand)
	return cost_func,grand

#数据准备
x=[[1,2,3,4,5,7,9,11],[4,2,5,3,6,7,5,7]]
#x=[[1,1,4],[1,2,2],[1,3,5],[1,4,3],[1,5,6],[1,7,7],[1,9,5],[1,11,7]]
x1=[[1,2,3,4],[4,2,5,3]]
x2=[[5,7,9,11],[6,7,5,7]]
y=[[-1,-1,-1,-1,1,1,1,1]]
#y=[-1,-1,-1,-1,1,1,1,1]

x_data=np.array(x)
x1_data=np.array(x1)
x2_data=np.array(x2)
y_data=np.array(y)
one_data=np.ones((y_data.shape[0],1))
#print(x1_data[0,:].shape)

#plf.scatter(x1_data[0,:],x1_data[1,:],marker='o')
#plf.scatter(x2_data[0,:],x2_data[1,:],marker='x')
#plf.show()
X=x_data.T
#X=np.insert(x_data.T,0,values=one_data,axis=1)
Y=y_data.T
#print(X)


#参数设置

sigma=0.3

theta=6*np.random.random((X.shape[1],1))

theta[0,:]=1

#theta=theta/np.sqrt(np.dot(theta.T,theta)-1)

c_para=1.5

J_cost=6

threshold=0.1

alpha=0.002   #学习率

#使用scipy.optimize的最小化求解方法求解 func函数

#cost_func,grand=cost_func_kf(theta,X,Y,c_para)

#print(cost_func,grand)

#res=opt.minimize(cost_func_kf,theta,args=(X,Y,c_para))

#result = opt.minimize(fun=cost_func_kf, x0=theta, args=(X,Y,c_para), method='TNC')
#print(result.cost_func,result.grand)
#对于使用scipy中的目标函数最小化的接口不是很熟悉，以后再使用该api进行目标函数的优化

#使用sklearn中的api进行训练

#svm_model=svm.SVC(C=1.5,max_iter=1000,kernel='linear',gamma='scale')
#svm_model=svm.SVC(kernel='linear')
#svm_model=svm.SVC(kernel='sigmoid')
svm_model=svm.LinearSVC()
svm_model.fit(X,Y)

#print(svm_model.coef_)
#theta=svm_model.coef_
#print(svm_model.support_vectors_)
#print(svm_model.support_)
#print(svm_model.)
print(svm_model.coef_,svm_model.intercept_)
#theta=svm_model.support_vectors_[1,:]
w=svm_model.coef_
b=svm_model.intercept_

#使用权值画出间隔线		
x_test=[1,2,3,4,5,6,7,8,9,10]
#y_test=(-theta[:,1]*np.array(x_test)-theta[:,0])/theta[:,2]
y_test=(-w[:,0]*np.array(x_test)-b[0])/w[:,1]
plf.scatter(np.array(x_test),y_test,marker='^')
plf.scatter(x1_data[0,:],x1_data[1,:],marker='o')
plf.scatter(x2_data[0,:],x2_data[1,:],marker='x')
plf.show()
	




