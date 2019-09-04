#coding=gbk

import matplotlib.pyplot as plf
import pandas as pd
import numpy as np
import os
import requests
from collections import deque


class stack():
    def __init__(self):
        self.stack=[]
    def push(self,data):
        self.stack.append(data)
        return self.stack
    def pop(self):
        if self.stack:
            return self.stack.pop()
        else:
            raise LookupError('stack is empty')
    def isempty(self):
        return bool(self.stack)
    def top(self):
        if self.stack:
            return self.stack[-1]
        else:
            print("stack is emspy")
            return None

if __name__=='__main__':
    stack_test=stack()
    stack_test.push(1)
    print(stack_test.pop())
