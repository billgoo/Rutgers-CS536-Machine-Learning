# -*- coding:utf-8 -*-
# Filename: train2.1.py
# Author：hankcs
# Date: 2015/1/30 16:29
import copy
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np


class Perceptron:

    def __init__(self, m, k, data):
        self.m = m
        self.k = k
        self.w = np.array([0] * (k + 1), dtype=np.float64)
        self.data = data
    
    
    def update(self, item):
        """
        update parameters using stochastic gradient descent
        :param item: an item which is classified into wrong class
        :return: nothing
        """
        self.w = self.w + item[-1] * item[0:self.k+1]
    
    
    def check(self, item):
        """
        check if the hyperplane can classify the examples correctly
        :return: true if it can
        """
        result = item[-1] * sum(self.w * item[0:self.k+1])
        if result > 0:
            return True
        else:
            return False
        return False

    def train(self):
        '''
        train the model
        :return w and b
        '''
        flag = True # True if not classify
        count = 0
        # print('-'*30)
        while flag:
            for i in range(0, self.m):
                if not self.check(self.data[i]):
                    count += 1
                    #print('No.{0} adjustment...'.format(count))
                    #print('data:'+str(self.data[i]))
                    #print('(w,b)=:'+str(self.w)+' '+str(self.b))
                    self.update(self.data[i])
                    flag = True
                    break
                else:
                    flag = False
                    # if all have been classified
            if count >= 1000000:
                # in case for infinity loop
                break
        # return model
        return (self.w, count)
 
 
if __name__ == "__main__":
    '''
    for i in range(1000):
        if not check(): break
 
    # first set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], 'g', lw=2)
    label = ax.text([], [], '')
 
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        x, y, x_, y_ = [], [], [], []
        for p in training_set:
            if p[1] > 0:
                x.append(p[0][0])
                y.append(p[0][1])
            else:
                x_.append(p[0][0])
                y_.append(p[0][1])
 
        plt.plot(x, y, 'bo', x_, y_, 'rx')
        plt.axis([-6, 6, -6, 6])
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Perceptron Algorithm (www.hankcs.com)')
        return line, label
 
 
    # animation function.  this is called sequentially
    def animate(i):
        global history, ax, line, label
 
        w = history[i][0]
        b = history[i][1]
        if w[1] == 0: return line, label
        x1 = -7
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 7
        y2 = -(b + w[0] * x2) / w[1]
        line.set_data([x1, x2], [y1, y2])
        x1 = 0
        y1 = -(b + w[0] * x1) / w[1]
        label.set_text(history[i])
        label.set_position([x1, y1])
        return line, label
 
    # call the animator.  blit=true means only re-draw the parts that have changed.
    print history
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=1000, repeat=True,
                                   blit=True)
    plt.show()
    anim.save('perceptron.gif', fps=2, writer='imagemagick')
    '''