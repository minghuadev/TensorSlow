#!/usr/bin/env python3

# code from the jupyter_notebook

import numpy as np
import matplotlib.pyplot as plt
import threading

class GetKeyKit(threading.Thread):
    def __init__(self):
        super().__init__()
        self._key_hit_count = 0
        self._stop_request = False
    def getHitCount(self):
        return self._key_hit_count
    def run(self):
        while self._stop_request != True:
            k = input("  Hit <entery> key ...")
            self._key_hit_count += 1
            print("  Hit count : ", self._key_hit_count)
        print("Key input thread finished\n")
    def set_stop(self):
        self._stop_request = True
        print("\nYou may need to hit <enter> key to stop ...\n")

# Create red points centered at (-2, -2)
red_points = np.random.randn(50, 2) - 2*np.ones((50, 2))

# Create blue points centered at (2, 2)
blue_points = np.random.randn(50, 2) + 2*np.ones((50, 2))

k_thread = GetKeyKit()
k_thread.start()

# Plot them
plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue')
plt.show(block=False)

def wait_key():
    k_cnt_prev = k_thread.getHitCount()
    while True:
        plt.pause(1) # 1 seconds, I use 1 usually
        k_cnt = k_thread.getHitCount()
        if k_cnt != k_cnt_prev:
            break
    plt.close("all")
wait_key()

# Plot a line y = -x
x_axis = np.linspace(-4, 4, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)

# Add the red and blue points
plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue')
plt.show(block=False)

wait_key()

# Create an interval from -5 to 5 in steps of 0.01
a = np.arange(-5, 5, 0.01)

# Compute corresponding sigmoid function values
s = 1 / (1 + np.exp(-a))

# Plot them
plt.plot(a, s)
plt.grid(True)
plt.show(block=False)

wait_key()

k_thread.set_stop()

