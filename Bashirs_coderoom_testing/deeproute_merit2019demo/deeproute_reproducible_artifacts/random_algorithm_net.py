#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 08:47:23 2019

@author: bashirm
"""

# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Networkpath_rand.csv')

# Implementing Random Selection
import random
N = 10000
d = 4
path_selected = []
total_reward = 0
for n in range(0, N):
    pt = random.randrange(d)
    path_selected.append(pt)
    reward = dataset.values[n, pt]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(path_selected)
plt.title('Random Algorithm path selections')
plt.xlabel('Network Path')
plt.ylabel('Number of times each path was selected')
plt.show()