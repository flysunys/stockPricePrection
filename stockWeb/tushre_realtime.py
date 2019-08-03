#coding:utf8

import tushare as ts
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plf
from matplotlib.pyplot import MultipleLocator

stockcode='601009'
#df = ts.get_realtime_quotes('601009')
endtime_str='2019-08-03 18:00:00'
endtime=datetime.datetime.strptime(endtime_str,"%Y-%m-%d %H:%M:%S")
#print(endtime)
#print("the type of endtime is : %s" % type(endtime))
starttime_ori=datetime.datetime.now()
#print(type(starttime_ori))
starttime_str=starttime_ori.strftime("%Y-%m-%d %H:%M:%S")
starttime=datetime.datetime.strptime(starttime_str,"%Y-%m-%d %H:%M:%S")
#print(starttime)
delta=endtime-starttime
t=[]
p=[]
while delta > datetime.timedelta(seconds=1):
	t.append(starttime)
	df = ts.get_realtime_quotes('601009')
	p.append(df['price'].values[0])
	plf.plot(t,p,'-r')
	plf.title('%s realtime price figure' % stockcode)
	plf.xlabel('time')
	plf.ylabel('price')
	plf.draw()
	plf.pause(2)
	#plf.show()
	starttime_ori=datetime.datetime.now()
	starttime_str=starttime_ori.strftime("%Y-%m-%d %H:%M:%S")
	starttime=datetime.datetime.strptime(starttime_str,"%Y-%m-%d %H:%M:%S")
	delta=endtime-starttime

