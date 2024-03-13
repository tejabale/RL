import numpy as np


temp = dict()

a = "hello"
if a not in temp.keys():
    temp[a] = 1
else:
    temp[a] = 0
print(temp)