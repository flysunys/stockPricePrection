import pymysql
import datetime
import re

class DBStock:
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
	def exit_of_table(table_name):
		conn = pymysql.connect(host="localhost",user="root",password="1234",database="test",charset="utf8")
		cursor=conn.cursor()
		sql = "SHOW TABLES"
		cursor.execute(sql)
		conn.commit()
		table_list = cursor.fetchall()
		cursor.close()
		conn.close()
		#table_list=re.findall('(\'.*?\')',str(tables))
		for tableone in table_list:
			if tableone == table_name:
				print("exits table %s" % table_name)
				return 1
			else:
				print("not exits table %s" % table_name)
				return 0
	def insert_of_table(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj):
		conn = pymysql.connect(host="localhost",user="root",password="1234",database="test",charset="utf8")
		cursor=conn.cursor()

		sql="Insert INTO `StockDataBackup` (`StockDate`,`StockCode`,`StockHigh`,`StockLow`,`StockOpen`,`StockClose`,`StockVolume`,`StockAdj`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"

		cursor.execute(sql,(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj))
		
		conn.commit()
		result = cursor.fetchone()
		cursor.close()
		conn.close()
		#result = cursor.fetchone()
		return result
		

