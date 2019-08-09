import pymysql
import datetime
import re

class DBDataNet:
	def establishTable():
		conn = pymysql.connect(host="localhost",user="root",password="1234",database="test",charset="utf8")
		cursor=conn.cursor()
		
		sql = "create table `NetDataStock`(\
	   `StockDate` date NOT NULL ,\
	   `StockCode` int(6) NOT NULL ,\
	   `StockHigh` double NOT NULL ,\
	   `StockLow` double NOT NULL ,\
	   `StockOpen` double NOT NULL ,\
	   `StockClose` double NOT NULL ,\
	   `StockVolume` double NOT NULL ,\
	   `StockAdj` double NOT NULL ,\
	   `Label` varchar(12) ,\
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
		count_table=0
		for tableone in table_list:
			print(tableone[0])
			#print(type(tableone))
			if tableone[0] == table_name:
				count_table = count_table + 1
		if count_table == 1:
			print("exits table %s" % table_name)
			return 1
		else:
			print("not exits table %s" % table_name)
			return 0
	def query_of_table(stockdate,stockcode):
		conn = pymysql.connect(host="localhost",user="root",password="1234",database="test",charset="utf8")
		cursor=conn.cursor()
		sql="SELECT * FROM `NetDataStock` WHERE `StockDate` = %s AND `StockCode` = %s"

		cursor.execute(sql,(stockdate,stockcode))
		conn.commit()
		result = cursor.fetchall()
		cursor.close()
		conn.close()
		return result
	def insert_of_table(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj):
		conn = pymysql.connect(host="localhost",user="root",password="1234",database="test",charset="utf8")
		cursor=conn.cursor()

		sql="Insert INTO `NetDataStock` (`StockDate`,`StockCode`,`StockHigh`,`StockLow`,`StockOpen`,`StockClose`,`StockVolume`,`StockAdj`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"

		cursor.execute(sql,(stockdate,stockcode,stockhigh,stocklow,stockopen,stockclose,stockvolume,stockadj))
		
		conn.commit()
		result = cursor.fetchone()
		cursor.close()
		conn.close()
		#result = cursor.fetchone()
		return result