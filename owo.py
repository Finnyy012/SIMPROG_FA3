import math
import mesa
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labels = [2,3,4,5,6,7,8,9,10]
y_p = [100,75,69,43,45,40,33,23,10]
y_r = [100,69,55,35,17,16,14,10,5]
y_a = [100,52,26,20,10,9,8,5,2]





x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, y_p, width, label='plurality')
rects2 = ax.bar(x,         y_r, width, label='instant runoff')
rects3 = ax.bar(x + width, y_a, width, label='approval')

ax.set_xlabel('n parties')
ax.set_ylabel('% of elections resulting in condorcet winner')
ax.set_xticks(x, labels)
ax.legend()

fig.tight_layout()

plt.show()
