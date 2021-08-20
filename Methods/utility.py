import numpy as np
import random

def find_nearest(data, val):
    idx = (np.abs(data - val)).argmin()
    return data[idx]

def find_point(x, y, pts):
    return np.where((pts == (x, y)).all(axis=1))[0][0]

def createGoals(start=(0, 0), goal=(1,1), res=100, domain=(-1, 1), seed=1):
    i_s = 0
    i_g = 0
    random.seed(seed)

    steps = int(np.sqrt(res))
    d = size/steps
    x, y = np.linspace(0, size, steps), np.linspace(0, size, steps)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.c_[X.ravel(), Y.ravel()]

    if all(i is not None for i in start):
        s_x = find_nearest(x, start[0])
        s_y = find_nearest(y, start[1])
        i_s = find_point(s_x, s_y, points)
    else:
        i_s = random.randint(0, int(0.2*res))

    if all(i is not None for i in goal):
        g_x = find_nearest(x, goal[0])
        g_y = find_nearest(y, goal[1])
        i_g = find_point(g_x, g_y, points)
    else:
        i_g = random.randint(int(0.8*res), res-1)

    return i_s, i_g