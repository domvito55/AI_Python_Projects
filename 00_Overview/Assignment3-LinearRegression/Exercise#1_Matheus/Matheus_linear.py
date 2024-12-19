# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:47:11 2022

@author: Matheus Teixeira
301236904
Comp 237 - Sec. 002
Week 5 - Exercise 1
"""
import numpy as np
import matplotlib.pyplot as plt


#### b. #########################
np.random.seed(4)

#### a. #########################
x=np.random.uniform(-1,1,100)

#### c. #########################
y = 12*x - 4

#### d. #########################
plt.scatter(x,y, alpha=0.5)
plt.title('y = 12x - 4')
plt.xlabel('x')
plt.ylabel('y')

#### e. #########################
noise = np.random.normal(size=100)
y = 12*x - 4 + noise

#### f. #########################
plt.scatter(x,y, alpha=0.5)
plt.title('y = 12x - 4 + noise')
plt.xlabel('x')
plt.ylabel('y')


