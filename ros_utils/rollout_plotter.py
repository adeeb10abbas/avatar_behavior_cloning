#!/usr/bin/env python

import pickle
from matplotlib import pyplot as plt
import numpy as np

with open("inferred_actions.pkl", "rb") as f:
    data = pickle.load(f)
    
print(data['left_gripper'].shape)
print(data['ground_truth'].shape)
# Plot the inferred actions

data['right_gripper'] = data['right_gripper'].reshape(-1, 3)
data['left_gripper'] = data['left_gripper'].reshape(-1, 3)
data['left_arm'] = data['left_arm'].reshape(-1, 9)
data['right_arm'] = data['right_arm'].reshape(-1, 9)

plt.figure()
ax1 = plt.subplot(411)
ax1.plot(data['right_gripper'], label='Right Gripper')
ax1.plot(data['ground_truth'][:data['right_gripper'].shape[0], :3], label='Ground Truth')

ax2 = plt.subplot(412)
ax2.plot(data['left_gripper'], label='Left Gripper')
ax2.plot(data['ground_truth'][:data['left_gripper'].shape[0], 12:15], label='Ground Truth')

ax3 = plt.subplot(413)
ax3.plot(data['left_arm'][:,0:3], label='Left Arm')
ax3.plot(data['ground_truth'][:data['left_arm'].shape[0], 15:18], label='Ground Truth')

ax4 = plt.subplot(414)
ax4.plot(data['right_arm'][:,0:3], label='Right Arm')
ax4.plot(data['ground_truth'][:data['right_arm'].shape[0], 3:6], label='Ground Truth')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()

