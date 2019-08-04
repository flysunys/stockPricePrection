#coding:utf8

import tushare as ts
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plf
from matplotlib.pyplot import MultipleLocator

#print([[1,1],[2,2],[3,3]])
#df = ts.shibor_ma_data()
df = ts.get_hist_data('603993',start='2019-01-01',end='2019-01-12')
#print(type(df))
#result_shibors=df.values
#for each_day in df:
#	print(each_day_shibor)
#print(df-np.mean(df,axis=0))
print(np.mean(df,axis=0))