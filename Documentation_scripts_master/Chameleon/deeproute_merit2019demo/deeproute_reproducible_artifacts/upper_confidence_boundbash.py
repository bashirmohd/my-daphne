# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Networkpath_rand.csv')

# Implementing UCB
import math
N = 10000
d = 4
path_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    pt = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            pt = i
    path_selected.append(pt)
    numbers_of_selections[pt] = numbers_of_selections[pt] + 1
    reward = dataset.values[n, pt]
    sums_of_rewards[pt] = sums_of_rewards[pt] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(path_selected)
plt.title('Histogram of path selections')
plt.xlabel('Network Path')
plt.ylabel('Number of times each path was selected')
plt.show()