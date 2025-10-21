# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 22:36:23 2022

@author: domvito55
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 2022
@Author: Matheus Teixeira
Title: Lab 5
Neural network
"""
import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

#_________ EX. 1 _________#
########## 4. ##########
np.random.seed(1)
########## 1. ##########
########## 2. ##########
min_val = -.6
max_val = .6
data_size = 10

set1 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
set2 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
input_Matheus = np.concatenate((set1, set2),axis=1)
########## 3. ##########
output_Matheus = set1 + set2
########## 5. ##########
firstLayerNumberOfNeurons = 6
numberOfOutputs = 1
# pass min an max for each input column. In this case, we have
# 2 columns with the same range.
nn = nl.net.newff([[min_val, max_val], [min_val, max_val]],
                  [firstLayerNumberOfNeurons, numberOfOutputs])
########## 6. ##########
########## 7. ##########
# Train the neural network
error_progress = nn.train(input_Matheus,
                          output_Matheus,
                          show=15,
                          goal=0.00001)
########## 8. ##########
result1 = nn.sim(np.array([.1,.2]).reshape(1,2))
result1b = nn.sim(np.array([.6,.6]).reshape(1,2))
original = nn.sim(input_Matheus)

def f(x, y):
    return (x + y)
x = np.linspace(-.6, .6, 600)
y = np.linspace(-.6, .6, 600)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 100, cmap='binary')
x1 = x.reshape(600,1)
y1 = y.reshape(600,1)
test_Matheus = np.concatenate((x1,y1),axis=1)
results = nn.sim(test_Matheus)
ax.scatter3D(x1.reshape(600,), y1.reshape(600,), results.reshape(600,))
ax.scatter3D(set1.reshape(data_size,), set2.reshape(data_size,), original.reshape(data_size,))
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x1 + x2')

#_________ EX. 2 _________#
########## 4. ##########
np.random.seed(1)
########## 1. ##########
########## 2. ##########
min_val = -.6
max_val = .6
data_size = 10

set1 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
set2 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
input_Matheus = np.concatenate((set1, set2),axis=1)
########## 3. ##########
output_Matheus = set1 + set2
########## 5. ##########
firstLayerNumberOfNeurons = 5
secondLayerNumberOfNeurons = 3
numberOfOutputs = 1
# pass min an max for each input column. In this case, we have
# 2 columns with the same range.
nn = nl.net.newff([[min_val, max_val], [min_val, max_val]],
                  [firstLayerNumberOfNeurons,
                   secondLayerNumberOfNeurons,
                   numberOfOutputs])
########## 6. ##########
########## 7. ##########
# Set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd
# Train the neural network
error_progress = nn.train(input_Matheus,
                          output_Matheus,
                          epochs=1000,
                          show=100,
                          goal=0.00001)
########## 8. ##########
result2 = nn.sim(np.array([.1,.2]).reshape(1,2))
result2b = nn.sim(np.array([.6,.6]).reshape(1,2))
original = nn.sim(input_Matheus)

x = np.linspace(-.6, .6, 600)
y = np.linspace(-.6, .6, 600)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 100, cmap='binary')
x1 = x.reshape(600,1)
y1 = y.reshape(600,1)
test_Matheus = np.concatenate((x1,y1),axis=1)
results = nn.sim(test_Matheus)
ax.scatter3D(x1.reshape(600,), y1.reshape(600,), results.reshape(600,))
ax.scatter3D(set1.reshape(data_size,), set2.reshape(data_size,), original.reshape(data_size,))
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x1 + x2')

#_________ EX. 3 _________#
########## 4. ##########
np.random.seed(1)
########## 1. ##########
########## 2. ##########
min_val = -.6
max_val = .6
data_size = 100

set1 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
set2 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
input_Matheus = np.concatenate((set1, set2),axis=1)
########## 3. ##########
output_Matheus = set1 + set2
########## 5. ##########
firstLayerNumberOfNeurons = 6
numberOfOutputs = 1
# pass min an max for each input column. In this case, we have
# 2 columns with the same range.
nn = nl.net.newff([[min_val, max_val], [min_val, max_val]],
                  [firstLayerNumberOfNeurons, numberOfOutputs])
########## 6. ##########
########## 7. ##########
# Train the neural network
error_progress = nn.train(input_Matheus,
                          output_Matheus,
                          show=15,
                          goal=0.00001)
########## 8. ##########
result3 = nn.sim(np.array([.1,.2]).reshape(1,2))
result3b = nn.sim(np.array([.6,.6]).reshape(1,2))
original = nn.sim(input_Matheus)

x = np.linspace(-.6, .6, 600)
y = np.linspace(-.6, .6, 600)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 100, cmap='binary')
x1 = x.reshape(600,1)
y1 = y.reshape(600,1)
test_Matheus = np.concatenate((x1,y1),axis=1)
results = nn.sim(test_Matheus)
ax.scatter3D(x1.reshape(600,), y1.reshape(600,), results.reshape(600,))
ax.scatter3D(set1.reshape(data_size,), set2.reshape(data_size,), original.reshape(data_size,))
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x1 + x2')

#_________ EX. 4 _________#
########## 4. ##########
np.random.seed(1)
########## 1. ##########
########## 2. ##########
min_val = -.6
max_val = .6
data_size = 100

set1 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
set2 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
input_Matheus = np.concatenate((set1, set2),axis=1)
########## 3. ##########
output_Matheus = set1 + set2
########## 5. ##########
firstLayerNumberOfNeurons = 5
secondLayerNumberOfNeurons = 3
numberOfOutputs = 1
# pass min an max for each input column. In this case, we have
# 2 columns with the same range.
nn = nl.net.newff([[min_val, max_val], [min_val, max_val]],
                  [firstLayerNumberOfNeurons,
                   secondLayerNumberOfNeurons,
                   numberOfOutputs])
########## 6. ##########
########## 7. ##########
# Set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd
# Train the neural network
error_progress = nn.train(input_Matheus,
                          output_Matheus,
                          epochs=1000,
                          show=100,
                          goal=0.00001)
########## 4.7 - Plot the training progress
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()

########## 8. ##########
result4 = nn.sim(np.array([.1,.2]).reshape(1,2))
result4b = nn.sim(np.array([.6,.6]).reshape(1,2))
original = nn.sim(input_Matheus)

x = np.linspace(-.6, .6, 600)
y = np.linspace(-.6, .6, 600)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 100, cmap='binary')
x1 = x.reshape(600,1)
y1 = y.reshape(600,1)
test_Matheus = np.concatenate((x1,y1),axis=1)
results = nn.sim(test_Matheus)
ax.scatter3D(x1.reshape(600,), y1.reshape(600,), results.reshape(600,))
ax.scatter3D(set1.reshape(data_size,), set2.reshape(data_size,), original.reshape(data_size,))
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x1 + x2')

#_________ EX. 5 _________#
########## 4. ##########
np.random.seed(1)
########## 1. ##########
########## 2. ##########
min_val = -.6
max_val = .6
data_size = 10

set1 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
set2 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
set3 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
input_Matheus = np.concatenate((set1, set2, set3),axis=1)
########## 3. ##########
output_Matheus = set1 + set2 + set3
########## 5. ##########
firstLayerNumberOfNeurons = 6
numberOfOutputs = 1
# pass min an max for each input column. In this case, we have
# 3 columns with the same range.
nn = nl.net.newff([[min_val, max_val], [min_val, max_val], [min_val, max_val]],
                  [firstLayerNumberOfNeurons, numberOfOutputs])
########## 6. ##########
########## 7. ##########
# Train the neural network
error_progress = nn.train(input_Matheus,
                          output_Matheus,
                          show=15,
                          goal=0.00001)
########## 8. ##########
result5 = nn.sim(np.array([.2,.1,.2]).reshape(1,3))
result5b = nn.sim(np.array([.6,.6,.6]).reshape(1,3))

#________________5.11________________#

########## 4. ##########
np.random.seed(1)
########## 1. ##########
########## 2. ##########
min_val = -.6
max_val = .6
data_size = 100

set1 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
set2 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
set3 = np.random.uniform(min_val, max_val, data_size).reshape(data_size,1)
input_Matheus = np.concatenate((set1, set2, set3),axis=1)
########## 3. ##########
output_Matheus = set1 + set2 + set3
########## 5. ##########
firstLayerNumberOfNeurons = 5
secondLayerNumberOfNeurons = 3
numberOfOutputs = 1
# pass min an max for each input column. In this case, we have
# 3 columns with the same range.
nn = nl.net.newff([[min_val, max_val], [min_val, max_val], [min_val, max_val]],
                  [firstLayerNumberOfNeurons, numberOfOutputs])
########## 6. ##########
########## 7. ##########
# Set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd
# Train the neural network
error_progress = nn.train(input_Matheus,
                          output_Matheus,
                          epochs=1000,
                          show=100,
                          goal=0.00001)
########## 4.7 - Plot the training progress
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()

########## 8. ##########
result6 = nn.sim(np.array([.2,.1,.2]).reshape(1,3))
result6b = nn.sim(np.array([.6,.6,.6]).reshape(1,3))
