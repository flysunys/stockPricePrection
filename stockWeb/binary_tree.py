#coding=gbk

import matplotlib.pyplot as plf
import pandas as pd
import numpy as np
import os
import requests


class Node():
    def __init__(self,data,lchild=None,rchild=None,depth=0):
        self.data=data
        self.depth=depth
        self.lchild=lchild
        self.rchild=rchild
    def create_two_tree(self,dataset,depth=0):
        if len(dataset)>0:
            m=len(dataset)
            #sortdataset=dataset(np.argsort(dataset))
            index_sort=dataset.argsort()
            #print(index_sort)
            sortdataset=[ dataset[index_sort[i]] for i in range(len(index_sort))]
            median=m//2
            self.data=sortdataset[median]
            self.depth=depth
            print(self.data,self.depth)
            plf.scatter(self.data,9-self.depth,label=str(self.data))
            self.lchild=self.create_two_tree(np.array(sortdataset[0:median]),depth=depth+1)
            self.rchild=self.create_two_tree(np.array(sortdataset[median+1:]),depth=depth+1)
            return self
        else:
            return None
            
            
            

class binary_tree(object):
    def __init__(self):
        self.c=0.02
        self.alpha=0.03
    def get_data(self):
        r_data=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
        print(r_data)
        with open('./data/iris.data','w') as f:
            f.write(r_data.text)
        #os.chdir('./data/')
        df=pd.read_csv('./data/iris.data',names=['sepal length','sepal width','petal length','petal width','class'])
        return df
if __name__=='__main__':
    one_instance=binary_tree()
    df_one=one_instance.get_data()
    test_data=df_one['sepal length'].values
    two_instance=Node(test_data)
    two_tree=two_instance.create_two_tree(test_data)
    plf.show()
    #print(two_tree.data,two_tree.depth)
    #print(len(test_data))
    #print(type(test_data))
    #print(df_one)
    #print(df_one['sepal length'])
    #print(one_instance.c)
    #print(df_one.describe)
    #df_one['sepal length'].hist()
    #plf.plot(df_one['sepal length'].values)
    
    
        


