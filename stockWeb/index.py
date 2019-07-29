from flask import Flask,render_template, request
import pymysql
import pandas_datareader.data as web
import datetime
from os import path as pth
p=pth.split(pth.realpath('G:\gitdir\gitprojects\stockPricePrection\stockWeb\DButils\dbstock.py'))[0]
pth.sys.path.append(p)
#sys.path.append('G:\gitdir\gitprojects\stockPricePrection\stockWeb\DButils\dbstock.py')
from dbstock import DBStock
app=Flask(__name__)

def insertalltotable(stock,stock_code):
	for x1,x2,x3,x4,x5,x6,x7 in zip(stock.index,stock['High'].values,stock['Low'].values,stock['Open'].values,stock['Close'].values,stock['Volume'].values,stock['Adj Close'].values):
		stockdate=x1.date()
		stockcode=stock_code
		stockhigh=float(round(x2,2))
		stocklow=float(round(x3,2))
		stockopen=float(round(x4,2))
		stockclose=float(round(x5,2))
		stockvolume=float(round(x6,2))
		stockadj=float(round(x7,2))
		#print(type(stockdate))
		#print(type(stockhigh))
		#print(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj)
		DBStock.insert_of_table(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj)

def dateRange(beginDate, endDate):
    dates = []
    dt = datetime.datetime.strptime(beginDate, "%Y-%m-%d")
    date = beginDate[:]
    while date <= endDate:
        dates.append(date)
        dt = dt + datetime.timedelta(1)
        date = dt.strftime("%Y-%m-%d")
    return dates

@app.route('/',methods=['GET','POST'])
def index():
	if request.method == "GET":
		print(request.args.get('start_time'))
		table_name="stockdatabackup"
		if DBStock.exit_of_table(table_name):
			print("存在此表")
		else:
			print("不存在此表")
			DBStock.establishTable()
		return render_template("index_start.html")
	if request.method == "POST":
		#print(request.form.get("start_time"))
		#request_info = request.values.to_dict();
		#print(request_info.get("start_time"))
		#if request.data == None:
		#	print("返回值为空")
		#else:
		#	print("返回值不为空")
		#print(request.json)
		#print(request.data)
		#print(request.form)
		#return redirect("/")
		starttime=request.form['start_time']
		endtime=request.form['end_time']
		stockcode=request.form['stock_code']
		#print(type(starttime))
		#print(type(stockcode))
		#date_starttime = datetime.datetime.strptime(starttime,'%Y-%m-%d').date()
		#date_endtime = datetime.datetime.strptime(endtime,'%Y-%m-%d').date()
		#stock_data_ss=web.DataReader(stockcode + '.SS','yahoo',date_starttime,date_endtime)
		#insertalltotable(stock_data_ss,stockcode)
		list_query_data=[]
		list_date=[]
		list_high=[]
		list_low=[]
		list_open=[]
		list_close=[]
		for date_day in dateRange(starttime,endtime):
			print(date_day)
			result_query=DBStock.query_of_table(date_day,stockcode)
			if DBStock.query_of_table(date_day,stockcode):
				print("exit %s %s" % (date_day,stockcode))
				list_query_data.append(result_query[0])
				list_date.append(datetime.datetime.strftime(result_query[0][0], "%Y-%m-%d"))
				list_high.append(result_query[0][2])
				list_low.append(result_query[0][3])
				list_open.append(result_query[0][4])
				list_close.append(result_query[0][5])
			else:
				print("not exit %s %s" % (date_day,stockcode))
			
				
		print(starttime,endtime,stockcode)
		print(list_date,list_high,list_low,list_open,list_close)
		
		return render_template("index_bak.html",list_query_data=list_query_data,list_date=list_date,list_high=list_high,list_low=list_low,list_open=list_open,list_close=list_close)
	return "请重新编写路由"
if __name__ == '__main__':
	app.run(debug=True)