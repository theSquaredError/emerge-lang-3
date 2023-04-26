# Script for buliding the spatial conceptualisation

import torch
import numpy as np
import random
import string


class World:
    '''
        Generates all the hidden details for the environment 
    
    
    '''
    def __init__(self, opts) -> None:
        # creating the radiuses of the concentric circles
        self.opts = opts
        self.radiuses = self.opts.radiuses
        self.generate_vertex()
        self.generate_colors()
        self.generate_vocabulary()
        self.generate_concepts()
        
    
    def generate_vertex(self):
        '''
        generating vertexes for the environment
        '''
        self.locations = (self.opts.max_DIMENSIONALITY - self.opts.min_DIMENSIONALITY)*\
                        torch.rand(self.opts.n_vertex, 2) + self.opts.min_DIMENSIONALITY
    
    
    
    def generate_colors(self):
        self.vertex_colors = np.random.choice(self.opts.n_colors,self.opts.n_vertex) + 201
    
    def generate_vocabulary(self):
        '''
        Returns vocabulary size equal to concept space size

        Arguments:
        ------------
                   n_concepts: number of concepts 
        Returns:
        ------------
                List containing vocabularies 
        '''
        self.vocabularies = []
        while(len(self.vocabularies)<self.opts.n_concepts):
            res = ''.join(random.choices(string.ascii_lowercase,k=3))
            if res not in self.vocabularies:
                self.vocabularies.append(res)


    def generate_concepts(self):
        '''
        Assigns concept for each concept in the concept space
        '''
        # concepts for sectors (starting from 1)
        self.sectors = [i for i in range(1,self.opts.n_sectors+1)]
        # concepts for segments
        self.segments = [i+100 for i in range(1, self.opts.n_segments+1)]
        # concepts for colors
        self.colors = [i+200 for i in range(1, self.opts.n_colors+1)]
        self.all_concepts = self.sectors + self.segments + self.colors

    def get_concepts(self, target_index: int, source_index: int) -> tuple:
        '''
        Gives the concepts of the corresponding source and target location

        Arguements:
        ----------------
                    target_loc : target location for the agent
                    source_loc : source location of the agent
        
        Returns:
        ----------------
                octant, segment ,quadrant
    
        '''
        target_loc = self.locations[target_index]
        source_loc = self.locations[source_index]
        co1 = target_loc[0] - source_loc[0]
        co2 = target_loc[1] - source_loc[1]
        point1, point2 = source_loc, target_loc
        length = torch.sqrt(torch.square(point1[0] - point2[0])+torch.square(point1[1] - point2[1]))
        angle = torch.rad2deg(torch.acos((point2[0]-point1[0])/length))
        if point1[1]>point2[1]:
            angle = 360 - angle
    
        octant = 0
        segment = 0
        quadrant = 0
        if angle>=0 and angle<=45: quadrant,octant = 1,1
        
        elif angle>45 and angle<=90: quadrant,octant = 1,2

        elif angle>90 and angle<=135: quadrant,octant = 2,3

        elif angle>135 and angle<=180: quadrant,octant = 2,4

        elif angle>180 and angle<=225: quadrant,octant =3,5
            
        elif angle>225 and angle<=270: quadrant,octant = 3,6

        elif angle>270 and angle<=315: quadrant,octant =4,7
        
        elif angle>315: quadrant,octant = 4,8
        
        
        # finding the circle
        # c_x,c_y =source_loc[0], source_loc[1] #coordinates of the origin
        distance = torch.sqrt(torch.square(co1) + torch.square(co2))
        
        # radiuses = torch.linspace(0, 20, steps=constants.n_segments)
        # radiuses = torch.load('data/radiuses.pt')

        for i, s in enumerate(self.radiuses):
            if distance<=s.item():
                segment = i
                break

        # color of target location
        color = self.vertex_colors[target_index]
        return quadrant, segment+101, quadrant+300, color


    def get_vocab_tensor(self, vocab):
        '''
            Create one hot encoding for the vocabulary given
        '''
        encoding = torch.zeros(len(self.vocabularies))
        i = self.vocabularies.index(vocab)
        encoding[i] = 1
        return encoding
    
    def get_concept_tensor(self, concept):
        '''
            Create one hot encoding for the concept given
        '''
        encoding = torch.zeros(len(self.all_concepts))
        i = self.all_concepts.index(concept)
        encoding[i] = 1
        return encoding
    def get_concept_enc(self, concept, concept_type, opts):
        '''
            Create one hot encoding for the concept given along with 
            one hot encoding for the order of the concept
            '''
        if concept_type == 'sector': 
            # o_e = opts.comm_encoding['sector']
            i = self.sectors.index(concept)
            c_e = [0]*len(self.sectors)
            c_e[i] = 1
            # c_e.extend(o_e)
            return c_e
        elif concept_type == 'segment':
            # o_e = opts.comm_encoding['segment']
            i = self.segments.index(concept)
            c_e = [0]*len(self.segments)
            c_e[i] = 1
            # c_e.extend(o_e)
            return c_e
        elif concept_type == 'color':
            # o_e = opts.comm_encoding['color']
            i = self.colors.index(concept)
            c_e = [0]*len(self.colors)
            c_e[i] = 1
            # c_e.extend(o_e)
            return c_e
        
        