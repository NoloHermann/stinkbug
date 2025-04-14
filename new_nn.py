# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:46:16 2022

@author: herma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

class dlnet:
    def __init__(self, x, y):
        self.X=x
        self.Y=y
        self.Yh=np.zeros((1,self.Y.shape[1]))
        self.L=2
        self.dims = [4, 7, 1]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.lr=0.0001
        self.sam = self.Y.shape[1]
        
    def nInit(self):    
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))                
        return
    
    def Sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
        
    def Relu(self, Z):
        return np.maximum(0,Z)
    
    def nloss(self,Yh):
        loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))    
        return loss
    
    def forward(self):    
        Z1 = self.param['W1'].dot(self.X) + self.param['b1'] 
        A1 = self.Relu(Z1)
        self.ch['Z1'],self.ch['A1']=Z1,A1
                
        Z2 = self.param['W2'].dot(A1) + self.param['b2']  
        A2 = self.Sigmoid(Z2)
        self.ch['Z2'],self.ch['A2']=Z2,A2
        self.Yh=A2
        loss=self.nloss(A2)
        return self.Yh, loss
        
    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    def dSigmoid(self, Z):
        s = 1/(1+np.exp(-Z))
        dZ = s * (1-s)
        return dZ
    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
        
        dLoss_Z2 = dLoss_Yh * self.dSigmoid(self.ch['Z2'])    
        dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
        dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                            
        dLoss_Z1 = dLoss_A1 * self.dRelu(self.ch['Z1'])        
        dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
        dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
        
        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
        
    def gd(self,X, Y, iter = 3000):
            np.random.seed(1)                         
        
            self.nInit()
        
            for i in range(0, iter):
                Yh, loss=self.forward()
                self.backward()
            
                if i % 500 == 0:
                    print ("Cost after iteration %i: %f" %(i, loss))
                    self.loss.append(loss)
        
            return
        
        
#################################################################################
f = open('bigData_log.txt','r')
#f = open('mixBoth_log.txt','r')

mq2 = []
mq5 = []
mq6 = []
mq135 = []
ir = []
rgb = []

for line in f:
    x = line.split(",")
    mq2.append(int(x[0]))
    mq5.append(int(x[1]))
    mq6.append(int(x[2]))
    mq135.append(int(x[3]))
    ir.append(int(x[4]))
    rgb.append(int(x[5]))
    
f.close()
 
for i in range(594-541):
    mq5[i+541] = mq5[i+540]-4
    
for i in range(594-440):
    mq2[i+440] = mq2[i+439]-4

targets = [0]
for i in range(10):
    targets.append(0)
for i in range(63-10):
    targets.append(1)
for i in range(127-63):
    targets.append(0)
for i in range(189-127):
    targets.append(1)
for i in range(238-189):
    targets.append(0)
for i in range(430-238):
    targets.append(0)
for i in range(559-430):
    targets.append(1)
for i in range(632-559):
    targets.append(0)
for i in range(687-632):
    targets.append(1)
for i in range(762-687):
    targets.append(0)
for i in range(606):
    targets.append(0)
for i in range(682-606):
    targets.append(1)
for i in range(748-682):
    targets.append(0)
for i in range(825-748):
    targets.append(1)
for i in range(883-825):
    targets.append(0)
for i in range(997-883):
    targets.append(1)
for i in range(1056-997):
    targets.append(0)

targets = np.array([targets])

input_vectors = np.array([mq2, mq5, mq6, mq135])

x = input_vectors[:, :-360]
y = targets[:, :-360]

nn = dlnet(x, y)
nn.gd(x, y, iter = 15000)

lost = np.array(nn.loss)
losts = []
for i in range(30):
    losts.append(lost[i,0,0])
    
plt.plot(losts[1:])
plt.title("Neural network error")
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.show()

print(nn.param)