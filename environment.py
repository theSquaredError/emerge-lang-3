# a script for 2D environment for multi-agent communication
import torch
import gym
import numpy as np
import random
import string
from typing import List
import networkx as nx
from node2vec import Node2Vec
import sys
from contextlib import contextmanager 
import os
from graphworld import World
@contextmanager 
def nullify_output(suppress_stdout=True,suppress_stderr=True):
    stdout = sys.stdout 
    stderr = sys.stderr 
    devnull = open(os.devnull, "w") 
    
    try: 
        if suppress_stdout: 
            sys.stdout = devnull 
        if suppress_stderr: 
            sys.stderr = devnull 
        yield 
    finally: 
        if suppress_stdout: 
            sys.stdout = stdout 
        if suppress_stderr: 
            sys.stderr = stderr


class Environment(gym.Env):
    def __init__(self,opts, world) -> None:
        super().__init__()
        self.opts = opts
        self.world = world
        self.obv_vec_size = opts.obv_vec_size
        self.locations = world.locations
        # self.action_space = spaces.Box()
        

        self.completeGraph = nx.complete_graph(self.locations.shape[0])
        # print(f'complete graph = {self.completeGraph}')
        for e in self.completeGraph.edges():
            source = self.locations[e[0]]
            target = self.locations[e[1]]
            length = torch.sqrt(torch.square(source[0] - target[0]) + torch.square(source[1]-target[1]))
            nx.set_edge_attributes(self.completeGraph,{e:{"weight":length}})
        with nullify_output():
            self.node2vec = Node2Vec(self.completeGraph,dimensions = 5, walk_length = 10, num_walks = 50, workers=1)
        self.model = self.node2vec.fit(window = 1, min_count = 0, batch_words = 1)
    
    
    
    
    def color_average(self, target_loc = None,agent_loc = None):
        # collect all the vertex in the region same as target location 
        # and average their color
        if target_loc == None:
            target_loc = self._target_location

        if agent_loc == None:
            agent_loc = self._agent_location
        C = []
        octant,segment, quadrant, color = self.world.get_concepts(self.target_index, self.source_index)
        for i,l in enumerate(self.locations):
            oct,seg,quad,color = self.world.get_concepts(i, self.source_index)
            if oct == octant and seg == segment:
                # append the colors
                color = self.opts.RGB_value[color -201] 
                C.append(color)
        C = torch.tensor(C)
        avg_color = torch.mean(C, dim=0)
        return avg_color
    
    
    def setObservation(self, center_index):
        # center = self.locations[center_index]
        s = self.model.wv.get_vector(center_index)
        
        return torch.tensor(s)
    
        
    def step(self,action):
        #

        pass
    
    def _get_obs(self,source_index = None,target_index = None):
        '''
            Return the observation dictionary for the given source and target location
            
            Parameters:
            --------------
                        source_index : index of the source location
                        target_index : index of the target location
        '''
        
        if source_index and target_index:
            source_obv = self.setObservation(source_index)
            target_obv = self.setObservation(target_index)
            return {"cur_loc": self.locations[source_index], "target": self.locations[target_index], \
            "agent_observation":source_obv, "target_embedding": target_obv, "avg_color": self.color_average(self.locations[target_index],self.locations[source_index] )}
        
        return {"cur_loc": self._agent_location, 
                "target": self._target_location, 
                "agent_observation":self.agent_observation, 
                "target_embedding": self.target_observation, 
                "avg_color": self.color_average(),
                }
    
    
    def reset(self):
        '''
        Resets agent and goal observation of the speaker and listener       
        
        '''
    
        self.source_index = np.random.choice(self.opts.n_vertex)
    
        self._agent_location = self.locations[self.source_index]
        
        # choosing target other than the agent location
        nonsrc_indices = [*range(self.opts.n_vertex)]
        nonsrc_indices.remove(self.source_index)
    
        self.target_index = np.random.choice(nonsrc_indices)
        self._target_location = self.locations[self.target_index]
        self.agent_observation = self.setObservation(self.source_index)
        self.target_observation = self.setObservation(self.target_index)

        self.world.get_concepts(self.target_index, self.source_index)

        observation = self._get_obs()
        
        return observation
