#coding:utf8

import tushare as ts
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plf
from matplotlib.pyplot import MultipleLocator


result=ts.get_hist_data('601009',start='2019-01-18',end='2019-07-31')
#print(result['close'].values)
close_list=result['close'].values
date_list=result.index.values
#close_list.to_list().reverse()
#print(close_list.to_list())
#print(type(close_list))
#print(list(close_list))
close_list_reverse=list(close_list)
close_list_reverse.reverse()
date_list_reverse=list(date_list)
date_list_reverse.reverse()
plf.figure(figsize=(10,5))
plf.plot(date_list_reverse,close_list_reverse)
plf.xlabel(s='date')
plf.ylabel(s='price')
plf.title('close price figure')
x_major_locator=MultipleLocator(20)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(1)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plf.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
#print(close_list_reverse)


volume_list=result['volume'].values
volume_list_reverse=list(volume_list)
volume_list_reverse.reverse()

plf.figure(figsize=(10,5))
plf.plot(volume_list_reverse)
plf.xlabel(s='date')
plf.ylabel(s='number')
plf.title('volume figure')
x_major_locator=MultipleLocator(20)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(100000)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plf.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plf.show()