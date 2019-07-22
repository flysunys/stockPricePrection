import pymysql
import pandas_datareader.data as web
import datetime

def establishTable():
	conn = pymysql.connect(host="localhost",user="root",password="1234",database="test",charset="utf8")
	cursor=conn.cursor()
	
	sql = "create table `StockDataBackup`(\
   `StockDate` date NOT NULL ,\
   `StockCode` int(6) NOT NULL ,\
   `StockHigh` double NOT NULL ,\
   `StockLow` double NOT NULL ,\
   `StockOpen` double NOT NULL ,\
   `StockClose` double NOT NULL ,\
   `StockVolume` double NOT NULL ,\
   `StockAdj` double NOT NULL ,\
   PRIMARY KEY (`StockDate`, `StockCode`))CHARSET=utf8"
	
	cursor.execute(sql)
	conn.commit()
	result = cursor.fetchone()
	cursor.close()
	conn.close()
	return result
	
	
	
def queryByname(name):
	conn = pymysql.connect(host="localhost",user="root",password="1234",database="test",charset="utf8")
	cursor=conn.cursor()

	sql="SELECT * FROM `student` WHERE `studentName` = %s"

	cursor.execute(sql,name)
	conn.commit()
	result = cursor.fetchone()
	cursor.close()
	conn.close()
	return result
def InsertData(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj):
	conn = pymysql.connect(host="localhost",user="root",password="1234",database="test",charset="utf8")
	cursor=conn.cursor()

	sql="Insert INTO `StockData` (`StockDate`,`StockCode`,`High`,`Low`,`StockOpen`,`StockClose`,`Volume`,`Adj`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"

	cursor.execute(sql,(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj))
	
	conn.commit()
	result = cursor.fetchone()
	cursor.close()
	conn.close()
	#result = cursor.fetchone()
	return result

start=datetime.date(2012,1,1)
end=datetime.date.today()

stock=web.DataReader('601009.SS','yahoo',start,end)
stock.dtype='float'
for x1,x2,x3,x4,x5,x6,x7 in zip(stock.index,stock['High'].values,stock['Low'].values,stock['Open'].values,stock['Close'].values,stock['Volume'].values,stock['Adj Close'].values):
	stockdate=x1.date()
	stockcode=601009
	stockhigh=float(round(x2,2))
	stocklow=float(round(x3,2))
	stockopen=float(round(x4,2))
	stockclose=float(round(x5,2))
	stockvolume=float(round(x6,2))
	stockadj=float(round(x7,2))
	#print(type(stockdate))
	#print(type(stockhigh))
	#print(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj)
	InsertData(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj)
