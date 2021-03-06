#coding:utf8

#@author:flysun
#@date:2019-10-08
#@last modified by:flysun
#@last modified time:2019-10-08

"""
description:

ucb-learning算法  强化学习

1、手动初始化Initialize Q(s,a) 和 count_value(s,a) 每个状态对应的每个action被选中的次数，初始化全为0

2、Repeat （for each episode）

3、初始化 环境状态  S

4、Repeat （for each step of episode）

5、Choose $a from $s using policy derived from $Q

计算ucb的值，ucb(s,a)=Q(s,a)/count_value(s,a)+math.sqrt(1/count_value(s,a))

对于当前状态，选择最大的ucb对应的action  进行执行；

更新count_value(s,a)：  count_value(s,a) += 1

6、Take Action $a   Observe $r  $s'

7、更新Q值，Q(s,a)=Q(s,a)+alpha*(r+gamma*max_{a'}Q(s',a')-Q(s,a))

8、更新进入下一个状态   $s = $s'

ucb-learning算法的说明：
	
	$s表示智能体agent所处的环境状态，$a为智能体所采取的动作，智能体采取动作之后会获得环境的反馈 $r
	
	以及观察到的新的环境状态 $s' ， $alpha为迭代步长，  $gamma 为折扣因子。取值范围为(0,1)，表达了
	
	对长远未来的考虑程度，当 $gamma = 0 时，只顾眼前的回报。 $Q 为选择动作时的 区别于$epsilion 贪婪策略（当 $rand() < $epsilion 时，选择具有最大 $Q(s,a) 值的动作 $a），
	
	选择的是最大ucb值对应的action
	
	$Q(s,a)为估计值，gamma*max_{a'}Q(s',a') 为实际值，实际上 gamma*max_{a'}Q(s',a') 也是估计值
	
	
	
"""



import time

import random

import os

import copy

import tkinter as tk

import math


####定义4X4的格子世界


class GridEnviroment():
	def __init__(self):
		self.n=4
		self.action=['East','South','West','North']
		self.terminal={'death':[{'rol':2,'col':1},{'rol':2,'col':2}],'target':[{'rol':3,'col':2}]}
		self.cav=None
		self.window=None
		
	def get_feed_back(self,state,last_state):
		rol=state['rol']
		col=state['col']
		r=0
		if last_state['rol']==rol and last_state['col']==col:
			return -1
		for t in self.terminal['death']:
			if rol==t['rol'] and col==t['col']:
				r=-1
		for t in self.terminal['target']:
			if rol==t['rol'] and col==t['col']:
				r=2
		return r
		
	def is_done(self,state):
		rol=state['rol']
		col=state['col']
		flag=False
		for t in self.terminal['death']:
			if rol==t['rol'] and col==t['col']:
				flag=True
				break
		for t in self.terminal['target']:
			if rol==t['rol'] and col==t['col']:
				flag=True
				break
		return flag
		
	def get_next_state(self,state,action):
		rol=state['rol']
		col=state['col']
		next_state={'rol':rol,'col':col}
		if action=='East':
			col+=1
		if action=='West':
			col-=1
		if action=='North':
			rol-=1
		if action=='South':
			rol+=1
		if rol < 0 or rol >= self.n:
			return next_state
		if col < 0 or col >= self.n:
			return next_state
		next_state['col']=col
		next_state['rol']=rol
		return next_state
		
	def init_env(self):
		self.window=tk.Tk()
		self.window.title('Grid World')
		self.window.geometry('300x300') #长和宽
		self.cav=tk.Canvas(self.window,bg='green',height=300,width=300)
	
	def reset_world(self):
		self.cav.delete()
		x0,y0=50,50
		step=50
		for rol in range(self.n):
			for col in range(self.n):
				self.cav.create_rectangle(x0+col*step,y0+rol*step,x0+(col+1)*step,y0+(rol+1)*step,fill='gray')
		for t in self.terminal['death']:
			rol=t['rol']
			col=t['col']
			self.cav.create_rectangle(x0+col*step,y0+rol*step,x0+(col+1)*step,y0+(rol+1)*step,fill='black')
		for t in self.terminal['target']:
			rol=t['rol']
			col=t['col']
			self.cav.create_rectangle(x0+col*step,y0+rol*step,x0+(col+1)*step,y0+(rol+1)*step,fill='yellow')
		rol=0
		col=0
		self.cav.create_rectangle(x0+col*step,y0+rol*step,x0+(col+1)*step,y0+(rol+1)*step,fill='red')
		self.cav.pack()
		
	def render_world(self,state):
		self.reset_world()
		x0,y0=50,50
		step=50
		rol=0
		col=0
		self.cav.create_rectangle(x0+col*step,y0+rol*step,x0+(col+1)*step,y0+(rol+1)*step,fill='gray')
		rol=state['rol']
		col=state['col']
		self.cav.create_rectangle(x0+col*step,y0+rol*step,x0+(col+1)*step,y0+(rol+1)*step,fill='red')
		self.cav.pack()
		self.window.update_idletasks()
		
#定义智能体		
		
class Agent():
	def __init__(self):
		self.init_state={'rol':0,'col':0}
		self.current_state={'rol':0,'col':0}
		self.current_action='East'
		self.action=['East','South','West','North']
		self.feed_back=0
		self.q_value={}
		self.count_value={}
		self.alpha=0.1
		self.epsilon=0.6
		self.gamma=0
	def reset_state(self):
		self.current_state={'rol':0,'col':0}
	def init_q_value(self,env):
		for rol in range(env.n):
			for col in range(env.n):
				state=str(rol)+'-'+str(col)
				self.q_value.update({state:{}})
				for a in self.action:
					self.q_value[state].update({a:0})
	def init_count_value(self,env):
		for rol in range(env.n):
			for col in range(env.n):
				state=str(rol)+'-'+str(col)
				self.count_value.update({state:{}})
				for a in self.action:
					self.count_value[state].update({a:0})
	def get_action_of_max_q_value(self,state):
		rol=state['rol']
		col=state['col']
		state=str(rol)+'-'+str(col)
		action_q_value=self.q_value[state]
		max_action=''
		max_q_value=-10000
		for action,q_value in action_q_value.items():
			if q_value > max_q_value:
				max_action=action
				max_q_value=q_value
		similar_max_action=[]
		for action,q_value in action_q_value.items():
			if q_value == max_q_value:
				similar_max_action.append(action)
		max_action=similar_max_action[random.randint(0,len(similar_max_action)-1)]
		return max_action,max_q_value
		
	def choose_action(self):
		rol=self.current_state['rol']
		col=self.current_state['col']
		state=str(rol)+'-'+str(col)
		action_q_value=self.q_value[state]
		action_count_value=self.count_value[state]
		ucb={}
		for action,count_value in action_count_value.items(): 
			if count_value != 0:
				ucb.update({action:1.0*action_q_value[action]/count_value+math.sqrt(1.0/count_value)})
			else:
				ucb.update({action:1.0*action_q_value[action]})
		maxactionvalue=-1
		max_action_d = self.current_action
		for action,ucb_value in ucb.items():
			if ucb_value > maxactionvalue:
				maxactionvalue = ucb_value
				max_action_d = action
		self.current_action = max_action_d
		next_state=env.get_next_state(self.current_state,self.current_action)
		if next_state == self.current_state:
			self.count_value[state].update({self.current_action:0})
		else:
			num_act = self.count_value[state][self.current_action] + 1
			self.count_value[state].update({self.current_action:num_act})
		
			
	def take_action(self,env):
		next_state=env.get_next_state(self.current_state,self.current_action)
		feed_back=env.get_feed_back(next_state,self.current_state)
		done=env.is_done(self.current_state)
		return next_state,feed_back,done
		
	def update_q_value(self,next_state,feed_back):
		rol=self.current_state['rol']
		col=self.current_state['col']
		current_state=str(rol)+'-'+str(col)
		q=self.q_value[current_state][self.current_action]
		_,max_q=self.get_action_of_max_q_value(next_state)
		q=q+self.alpha*(feed_back+self.gamma*max_q-q)
		self.q_value[current_state][self.current_action]=q
		
		
		
		
#使用Q-learning 寻宝

env=GridEnviroment()
agent=Agent()
agent.epsilion=0.8
agent.gamma=0.2
agent.init_q_value(env)
agent.init_count_value(env)
env.init_env()

max_episode=30

for episode in range(max_episode):
	agent.reset_state()
	print('---------------第 %d 个回合---------------' % episode)
	while True:
		env.render_world(agent.current_state)
		agent.choose_action()
		next_state,feed_back,done=agent.take_action(env)
		agent.update_q_value(next_state,feed_back)
		agent.current_state=next_state
		if done:
			break
			
print(agent.q_value)
env.window.mainloop()

		
		





