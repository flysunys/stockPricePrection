from flask import Flask,render_template, request
import pymysql
import pandas_datareader.data as web
import datetime
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
	if request.method == "GET":
		print(request.args.get('start_time'))
		return render_template("index_bak.html")
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
		print(starttime,endtime,stockcode)
		
		return '<h1>你好<h1>'
	return "请重新编写路由"
if __name__ == '__main__':
	app.run(debug=True)