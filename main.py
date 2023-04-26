""" ##Hyperparameters
* location_space: the number of vertexes in the 2D world
* n_sectors: numbers of sectors for the agent
* n_segments: number of segments for the agent
* n_colors: number of colors for the agent

"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from time import gmtime, strftime

from utils import DotDic
from pprint import pprint
from environment import Environment
from chatbots import  Team
import itertools
from train import train
import torch
from graphworld import World
import sys
import utils

from data_gen import DataLoader
# concept space is ordered as follows: [sector[0-7], segment[8-11], color[12-15]]

# sys.stdout = open("console_output.txt", "w")
order_vec = list(itertools.permutations(['segment', 'sector', 'color']))
opts={
    'n_agents':2,
    'n_vertex': 20,
    'uniAttrVal':12,
    'numAttr':3,
    'n_sectors': 4, 
    'n_segments': 4, 
    'n_colors': 4,
    'n_concepts':12,
    'n_vocab': 4,
    'obv_vec_size':13,
    'order_vec': order_vec,
    'order_vec_size' : len(order_vec),
    'trainSize':0.9,
    'imgFeatSize' : 20,
    'max_DIMENSIONALITY': 10,
    'min_DIMENSIONALITY': -10,
    'RGB_value' : [[0.6, 0.0, 0.3],
                [0.2, 0.2, 1.0],
                [0.1, 0.8, 0.4],
                [0.4, 0.8, 0.7]],
    
    
    'positve_reward': 1,
    'negative_reward': -1,
    'batchSize': 1000,

    'numEpochs': 1000000,
    'rlScale':100,
    'rnn_hidden_size': 128,

    'rnn_size':128,
    "learningRate" : 0.001,
    "momentum" : 0.05,
    "eps":0.05,
    "nepisodes":500,
}

radiuses = torch.linspace(0, 20, steps= opts['n_segments'])
opts['radiuses'] = radiuses
params = DotDic(opts)
data = DataLoader(DotDic(opts))
opts['data'] = data
world = World(DotDic(opts))

numInst = data.getInstCount()
utils.saveGraph(world=world, opts=DotDic(opts))

# pprint(opts)

# ---------------------------------------------------------------------
# Build agent, and setup optimiser
# ---------------------------------------------------------------------
team = Team(params)
optimizer = optim.Adam([{'params': team.speaker.parameters(),\
                         'lr': params['learningRate']},
                         {'params': team.listener.parameters(),\
                          'lr': params['learningRate']}])

# ---------------------------------------------------------------------
# Train the agents
# ---------------------------------------------------------------------

# begin training
numIterPerEpoch = int(np.ceil(numInst['train']/params['batchSize']))
numIterPerEpoch = max(1, numIterPerEpoch)
count = 0

matches = {}
accuracy = {}
best_accuracy = 0

for iterId in range(params['numEpochs'] * numIterPerEpoch):
    epoch = float(iterId) / numIterPerEpoch
    
    batchImg = data.getBatch(params['batchSize'])

    pred = team.forward(torch.Tensor(batchImg))

    # backward pass
    batchReward = team.performBackward(optimizer)


    optimizer.step()

    # # switch to evaluate
    # team.evaluate()
    # img = data.getCompleteData('train')
    # pred,_ = team.forward(Variable(img))
    pred = pred
    batchImg = batchImg.detach()
    # computer accuracy for segment, sector, color
    firstMatch = pred[:,0] == batchImg[:, 0]
    secondMatch = pred[:,1] == batchImg[:, 1]
    thirdMatch = pred[:,2] == batchImg[:, 2]
    matches['train'] = firstMatch & secondMatch & thirdMatch
    accuracy['train'] = 100 *torch.sum(matches['train'])\
                        / float(matches['train'].size(0))
    
    # switch to train

    if iterId % 100 != 0: continue

    time = strftime("%a, %d %b %Y %X", gmtime())
    print('[%s][Iter: %d][Ep: %.2f][R: %.4f][Tr: %.2f ]' % \
                                (time, iterId, epoch, team.totalReward,\
                                accuracy['train']))


# sys.stdout.close()



