#coding=gbk

import matplotlib.pyplot as plf
import pandas as pd
import numpy as np
import os
import requests
from collections import deque
import math

class DecisionTree():
    def __init__(self):
        self.threshold=0.2
        self.iters=1000
    def get_data(self):
        r_data=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
        print(r_data)
        with open('./data/adult.data','w') as f:
            f.write(r_data.text)
        #os.chdir('./data/')
        df=pd.read_csv('./data/adult.data',names=['age','workclass','fnlwgt','education','education-num',\
                       'marital-status','occupation','relationship','race','sex','capital-gain','capital-loss',\
                       'hours-per-week','native-country','income-class'])
        return df
        
    def calculate_shanno_Y(self,Y):
        num_Y=Y['income-class'].count()
        df_Y=Y.groupby(['income-class']).size()
        #df_Y=Y.groupby().size()
        #df_dict=df_Y.to_dict()
        df_list=df_Y.tolist()
        print(df_list)
        shanno_Y=0
        #for key_Y,value_Y in df_dict:
        #    value_Y
        for value_Y in df_list:
            p_Y=1.0*float(value_Y)/num_Y
            print(p_Y)
            shanno_Y+=p_Y*math.log(p_Y,2)
        return -shanno_Y
    def get_columns_data(self,data):
        return data.columns.values.tolist()
    def calculate_shanno_X(self,data,columns_x):
        H_shano_D=self.calculate_shanno_Y(data)
        H_shano_X=0
        num_X=data[columns_x].count()
        df_X=data.groupby([columns_x]).size()
        df_list=df_X.tolist()
        p_d=[]
        for value_x in df_list:
            p_x=1.0*float(value_x)/num_X
            #print(p_x)
            #shanno_Y+=p_Y*math.log(p_Y,2)
            p_d.append(p_x)
        h_d=[]
        for groupname,grouplist in data.groupby([columns_x]):
            h_d_each=self.calculate_shanno_Y(grouplist)
            h_d.append(h_d_each)
        shano_x=0
        for i,j in zip(p_d,h_d):
            shano_x+=i*j
        return shano_x
        
            
        
        
        
if __name__=='__main__':
    demo=DecisionTree()
    df_demo=demo.get_data()
    print(df_demo.columns.values.tolist())
    #print(df_demo['income-class'].count())
    #print(df_demo)
    #print(df_demo['age'])
    df_Y=df_demo.groupby(['income-class']).size()
    #for groupname,grouplist in df_demo.groupby(['income-class']):
    #    print(groupname)
    #    print(grouplist)
    #print(dir(df_Y),type(df_Y),df_Y)
    df_dict=df_Y.to_dict()
    H_D=demo.calculate_shanno_Y(df_demo)
    print(H_D)
    age_shano=demo.calculate_shanno_X(df_demo,'age')
    print(age_shano)
    
    #print(df_Y.tolist())
