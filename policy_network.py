# Script for the policy architecture of agent

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical

def cat_softmax(probs):
    cat_distr = OneHotCategorical(probs = probs)
    return cat_distr.sample(), cat_distr.entropy()


class Common(nn.Module):
    def __init__(self, params, input_size, hidden_size, output_size):
        super(Common, self).__init__()
        self.actions = []
        # absorb all the parameters to self
        for key, value in params.items():
            setattr(self, key, value)
    
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    
    def reinforce(self, rewards):
        for index, action in enumerate(self.actions):
            self.actions[index] = action * rewards
    
    def performBackward(self):
        sum([ii.sum() for ii in self.actions]).backward()

    
    def forward_(self, input, hidden):
        h1, c1 = self.lstm(input, hidden)
        out = self.out(h1)
        return out, h1, c1

    def initHidden_(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size)

#----------------------------------------------------------------------------------------
#                        Speaker 
#----------------------------------------------------------------------------------------
class spLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(spLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input, hidden):
        h1, c1 = self.lstm(input, hidden)
        out = self.out(h1)
        return out, h1, c1

    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class SNet(Common):
    '''
     Speaker network
     Takes the input as the concatenation of the attributes values 
     1. Encode the input vector to a hidden vector
     2. Pass the hidden vector to the LSTM
     3. LSTM input will be the output of the previous LSTM
     4. LSTM output will be the message
    '''
    def __init__(self, opts) -> None:
        self.parent = super(SNet, self)
        # super(SNet, self).__init__()
        self.opts = opts
        self.imgFeatSize = opts.imgFeatSize
        self.numAttr = opts.numAttr
        self.evalFlag = False
        self.output_size = opts.n_vocab
        self.input_size = opts.n_vocab
        # self.hidden_size = opts.rnn_hidden_size
        self.hidden_size = self.numAttr*self.imgFeatSize
        self.n_concepts = opts.n_concepts
        self.parent.__init__(opts,self.input_size, self.hidden_size, self.output_size)
        self.enc = nn.Embedding(self.n_concepts, self.imgFeatSize)

    
    def embedding(self, batch):
        embeds = self.enc(batch)
        features = embeds.view(embeds.shape[0], -1)
        return features
    
    '''def reinforce(self, rewards):
        for index, action in enumerate(self.actions):
            self.actions[index] = action * rewards'''
    
    # getting 3 messages from the speaker
    def forward(self, input):
        '''
        Returns the vocabulary chosen for each concept
        '''
        MSG_LEN = 3
        batch_size = input.shape[0]
        # hidden_s = self.spLSTM.initHidden(batch_size)
        cell_s = self.initHidden_(batch_size)
        message = []
        log_probs  = []
    
        # encode the input
        hidden_s = self.embedding(input)
        output_s = torch.zeros(batch_size, self.input_size)
        
        for i in range(MSG_LEN):
            output_s, hidden_s, cell_s = self.forward_(output_s, (hidden_s, cell_s))
            outDistr = F.softmax(output_s, dim=1)
            
            action_sampler = torch.distributions.Categorical(outDistr)
            actions = action_sampler.sample()
            # predict,entropy = cat_softmax(probs)
            log_prob= -action_sampler.log_prob(actions)
            
            # probs = F.softmax(output_s, dim=1)
            # predict, entropy = cat_softmax(probs)
            # log_prob = torch.log((probs*predict).sum(dim=1))
            self.actions.append(log_prob)
            message.append(outDistr)
        message = torch.stack(message)
        return message, actions
    
    # backward computation
    '''def performBackward(self):
        sum([ii.sum() for ii in self.actions]).backward()'''
    # switch mode to evaluate
    def evaluate(self):
        self.evalFlag = True

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

#------------------------------------------------------------------------------------------------------
#                       Listener
#------------------------------------------------------------------------------------------------------

class lisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(lisLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self,input, hidden):
        h1, c1 = self.lstm(input, hidden)
        out = self.out(h1)
        return out,h1, c1
    
    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class LNet(Common):
    def __init__(self, opts) -> None:
        self.opts = opts
        self.output_size = opts.uniAttrVal
        self.input_size = opts.n_vocab
        self.hidden_size = opts.rnn_hidden_size
        self.evalFlag = False
        super(LNet, self).__init__(opts,self.input_size, self.hidden_size, self.output_size)
    
    def forward(self, msg):
        pred_concepts = []
        log_probs = []
        entropy = 0.
        batch_size = msg.shape[1]
        hidden_l = self.initHidden_(batch_size)
        cell_l = self.initHidden_(batch_size)
        for i in range(msg.shape[0]):
            output, hidden_l, cell_l = self.forward_(msg[i], (hidden_l, cell_l))            
            outDistr = F.softmax(output, dim=1)
            action_sampler = torch.distributions.Categorical(outDistr)
            actions = action_sampler.sample()
            # predict,entropy = cat_softmax(probs)
            log_prob= -action_sampler.log_prob(actions)
            pred_concepts.append(actions)
            self.actions.append(log_prob)
        pred_concepts = torch.stack(pred_concepts, dim=1)
        # log_probs = torch.stack(log_probs, dim=1)
        return pred_concepts, log_probs
        
    '''def reinforce(self, rewards):
        for index, action in enumerate(self.actions):
            self.actions[index] = action * rewards'''
    # backward computation
    '''def performBackward(self):
        sum([ii.sum() for ii in self.actions]).backward()'''
    
    # switch mode to evaluate
    def evaluate(self):
        self.evalFlag = True

    def initHidden(self):
        return torch.zeros(self.hidden_size)


#---------------------------------------------------------------------------------   

class Team:
    def __init__(self, params) :
        self.params = params
        for field,value in params.items(): setattr(self,field,value)
        self.speaker = SNet(params)
        self.listener = LNet(params)
        self.reward = torch.Tensor(self.batchSize)
        self.totalReward = None
        self.rlNegReward = -10*self.rlScale
        # switch to train mode
    def train(self):
        self.speaker.train()
        self.listener.train()
    
    def evaluate(self):
        self.speaker.eval()
        self.listener.eval()

    
    def forward(self, batch):
        # intialize the hidden state of the speaker of the speaker as the features
        # talking round
        msg, log_probs = self.speaker(batch)
        # listening round
        msg = msg.detach()
        self.pred_concepts, self.predDistr = self.listener(msg)
        
        return self.pred_concepts, self.predDistr

    def backward(self, optimiser, batchLabels):
        # compute reward
        self.reward.fill_(self.rlNegReward)
        # all the three attributes needs to be predicted correctly

        first_match = self.pred_concepts[:,0] == batchLabels[:,0]
        second_match = self.pred_concepts[:,1] == batchLabels[:,1]
        third_match = self.pred_concepts[:,2] == batchLabels[:,2]
        
        self.reward[first_match & second_match & third_match] = self.rlScale

        # reinforce all the actions for speaker and listener bot

        self.speaker.reinforce(self.reward)
        self.listener.reinforce(self.reward)

            # optimise
        optimiser.zero_grad()  
        self.speaker.performBackward()
        self.listener.performBackward()

        # clip the gradient
        for p in self.speaker.parameters():
            p.grad.data.clamp_(min = -5, max = 5)
        for p in self.listener.parameters():
            p.grad.data.clamp_(min = -5, max = 5)
        
        # cumulative reward
        batchReward = torch.mean(self.reward)/self.rlScale
        if self.totalReward == None: self.totalReward = batchReward
        self.totalReward = 0.95 * self.totalReward + 0.05 * batchReward

        return batchReward

        
