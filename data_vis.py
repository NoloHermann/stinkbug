# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:19:05 2022

@author: herma
"""

import matplotlib.pyplot as plt
import numpy as np

f = open('bigData_log.txt','r')

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
    
plt.plot(mq2[-360:-180], label = "MQ2")
plt.plot(mq5[-360:-180], label = "MQ5")
plt.plot(mq6[-360:-180], label = "MQ6")
plt.plot(mq135[-360:-180], label = "MQ135")
plt.plot(ir[-360:-180], label = "IR")
plt.plot(rgb[-360:-180], label = "RGB")
#plt.plot(targets, label = "Targets")
plt.title('Gas sensor outputs')
plt.ylabel('ADC Value')
plt.xlabel('Samples')
plt.legend()
plt.show()

# index 0 = 0
# index 1->73 = 1
# index 74->110 = 0
# index 111->200 = 1
# index 201->237 = 0

targets = np.array([targets])
print(targets.shape)