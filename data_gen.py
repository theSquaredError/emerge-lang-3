# Generate the training data 
import functools
import itertools
from pprint import pprint
import random
import torch

class DataLoader:
    def __init__(self, params) :
        for field, value in params.items():
            setattr(self, field, value)
        self.createDataset()

    def getBatch(self, batchSize):
        # sample a batch
        indices = torch.LongTensor(batchSize).random_(0, self.numInst['train'])
        # print(f'indices: {indices}')
        batch = self.data_d['train'][indices]
        return batch
    
    def getInstCount(self): return self.numInst
    
    def getCompleteData(self, dtype):
        indices = torch.LongTensor(torch.arange(0, self.numInst[dtype]))
        return self.data_d[dtype][indices]
    def createDataset(self):
        # aOutVocab = [chr(ii + 65) for ii in range(params['aOutVocab'])]
        attributes = ['segments', 'sectors', 'colors']
        props = {'segments': ['seg1', 'seg2', 'seg3', 'seg4'],\
                    'sectors': ['sec1', 'sec2', 'sec3', 'sec4'],\
                    'colors': ['col1', 'col2', 'col3', 'col4']}
        
        attrList = [props[ii] for ii in attributes]
        dataVerbose = list(itertools.product(*attrList))
        numImgs = len(dataVerbose)
        self.numInst = {}
        self.numInst['train'] = int(self.trainSize * numImgs)
        self.numInst['test'] = numImgs - self.numInst['train']
        numAttrs = 3
        attrVals = ['seg1', 'seg2', 'seg3', 'seg4', 'sec1', 'sec2', 'sec3', 'sec4', 'col1', 'col2', 'col3', 'col4']
        # randomly select test
        splitData = {}
        splitData['test'] = random.sample(dataVerbose, self.numInst['test'])
        splitData['train'] = list(set(dataVerbose) - set(splitData['test']))
        self.attrVocab = {value: ii for ii, value in enumerate(attrVals)}
        self.invAttrVocab = {index: attr for attr, index in self.attrVocab.items()}
        self.data_d = {}
        for dtype in ['train', 'test']:
            data = torch.LongTensor(self.numInst[dtype], numAttrs)
            for ii, attrSet in enumerate(splitData[dtype]):
                data[ii] = torch.LongTensor([self.attrVocab[at] for at in attrSet])
                self.data_d[dtype] = data

