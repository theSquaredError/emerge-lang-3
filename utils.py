import copy
import torch
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from graphworld import World
import torch
from matplotlib.pyplot import fill
import networkx as nx
from node2vec import Node2Vec
import random
import matplotlib
import sys
matplotlib.use('Agg')



# Dictionary with items accessible as attributes (through '.' operator)
class DotDic(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __deepcopy__(self, memo=None):
		return DotDic(copy.deepcopy(dict(self), memo=memo))


def graphUtil(opts,locations, radiuses, origin, index):
    xs = [round(i[0].item(),3) for i in locations]
    ys = [round(i[1].item(),3) for i in locations]
    # radiuses = torch.linspace(0, 50, steps=20)
    # print(radiuses)
    xmin, xmax, ymin, ymax = -20, 20, -20, 20

    fig, ax  = plt.subplots(figsize=(15,15))
    ticks_frequency = 1
    ax.scatter(xs,ys)

    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')
    ax.spines['bottom'].set_position(('data',origin[1]))
    ax.spines['left'].set_position(('data', origin[0]))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.grid(which='both', color='grey', linewidth=1, linestyle='--', alpha=0.2)

    x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
    for i in range(len(xs)):
        # plt.annotate((xs[i],ys[i]), (xs[i], ys[i] + 0.2))
        plt.annotate( str(i),(xs[i],ys[i]) )
    ax.axline((origin[0],origin[1]), slope= math.tan(math.radians(45)), c='black' , linewidth = 1, linestyle = '-')
    ax.axline((origin[0],origin[1]), slope= math.tan(math.radians(135)),  c='black', linewidth = 1, linestyle = '--')
    ax.axline((origin[0],origin[1]), slope= math.tan(math.radians(0)), c='black', linewidth = 1, linestyle = '--')
    ax.axline((origin[0],origin[1]), slope= math.tan(math.radians(90)), c='black', linewidth = 1, linestyle = '--')
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])
    ax.set_xticks(np.arange(xmin, xmax+1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax+1), minor=True)
    for i in range(len(radiuses)):
        circle1 = plt.Circle(origin, radiuses[i],fill = False, ls = '--')
        ax.add_patch(circle1)
    

    # scatter plot
    # colors = [opts.RGB_value[0], opts.RGB_value[1]]
    # plt.scatter(xs,ys, c = colors, alpha = 0.9 , s=50)
    plt.scatter(xs,ys, alpha = 0.9 , s=50)
    plt.legend(['0','1','2','3'])
    

    plt.savefig('data/'+'plot_'+ str(index)+'.pdf')
    plt.close()
    # plt.show()


def saveGraph(opts,world):
    locations = world.locations
    radiuses = opts.radiuses
    torch.save(radiuses, 'data/radiuses.pt')
    torch.save(locations, 'data/locations.pt')
    f = open('data/locations_details.txt',"w")
    for i, loc in enumerate(locations):
        graphUtil(opts,locations, radiuses, loc, i)
        f.write(f"{i}: {loc}\n")
    f.close()
