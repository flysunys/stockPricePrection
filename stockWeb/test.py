num=input()
digital_list=input()
digital_num=[]
for each_num in digital_list.split(" "):
  digital_num.append(int(each_num))
max=max(digital_num)
min=min(digital_num)
digital_num.sort()
if int(num) % 2 == 0 and int(num) > 1:
  medium=round((digital_num[int(num)//2]+digital_num[(int(num)//2)-1])/2.0,2)
elif int(num) == 1:
  medium=max
else:
  medium=digital_num[(int(num)-1)//2]
print("%s %s %s" % (max,medium,min))