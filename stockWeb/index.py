from flask import Flask,render_template, request
import pymysql
import pandas_datareader.data as web
import datetime
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
	if request.method == "GET":
		return render_template("index_bak.html")
	if request.method == "POST":
		print(request.headers)
		request_info = request.values.to_dict();
		print(request_info.get("start_time"))
		if request.data == None:
			print("返回值为空")
		print(request.json)
		print(request.data)
		return redirect("/")
	return "请重新编写路由"
if __name__ == '__main__':
	app.run(debug=True)