import torch
import torch.nn as nn
import torch.nn.functional as F


class Speaker(nn.Module):
    def __init__(self, params):
        # absorb all the parameters to self
        super(Speaker, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)
        
        self.output_size = self.n_vocab
        self.input_size = self.n_vocab
        self.hidden_size = self.numAttr*self.imgFeatSize
        self.actions = []
        self.embed = nn.Embedding(self.n_concepts, self.imgFeatSize)

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def reinforce(self, rewards):
        for index, action in enumerate(self.actions):
            self.actions[index] = action * rewards
    
    
    def backward(self):
        sum([ii.sum() for ii in self.actions]).backward()
    
    def embedding(self, batch):
        embeds = self.embed(batch)
        features = embeds.view(embeds.shape[0], -1)
        return features

    def get_message(self,batch):
        
        MSG_LEN = 3
        cell_s = self.initHidden(self.batchSize)
        hidden_s = self.embedding(batch)
        output_s = torch.zeros(self.batchSize, self.input_size)
        message = []
        self.actions = []
        for i in range(MSG_LEN):
            hidden_s, cell_s = self.lstm(output_s, (hidden_s, cell_s))
            output_s = self.out(hidden_s)
            outDistr = F.softmax(output_s, dim=1)

            action_sampler = torch.distributions.Categorical(outDistr)
            actions = action_sampler.sample()
            log_prob= -action_sampler.log_prob(actions)

            self.actions.append(log_prob)
            message.append(outDistr)
        message = torch.stack(message)
        return message

    def initHidden(self,batchSize):
        return torch.zeros(batchSize, self.hidden_size)

class Listener(nn.Module):
    def __init__(self, params) -> None:
        super(Listener, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)
        self.input_size = self.n_vocab
        self.hidden_size = self.rnn_hidden_size
        self.output_size = self.uniAttrVal

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.actions = []

    def reinforce(self, rewards):
        for index, action in enumerate(self.actions):
            self.actions[index] = action * rewards

            
    def backward(self):
        sum([ii.sum() for ii in self.actions]).backward()
        
        
    def get_concepts(self, message):
        pred_concepts = []
        self.actions = []
        hidden_l = self.initHidden(self.batchSize)
        cell_l = self.initHidden(self.batchSize)
        for i in range(message.shape[0]):
            hidden_l, cell_l = self.lstm(message[i], (hidden_l, cell_l))
            output_l = self.out(hidden_l)
            outDistr = F.softmax(output_l, dim=1)
            action_sampler = torch.distributions.Categorical(outDistr)
            actions = action_sampler.sample()
            log_prob= -action_sampler.log_prob(actions)
            self.actions.append(log_prob)
            pred_concepts.append(actions)
        pred_concepts = torch.stack(pred_concepts,dim=1)
        return pred_concepts, outDistr

    def initHidden(self,batchSize):
        return torch.zeros(batchSize, self.hidden_size)
        



class Team:
    def __init__(self, params) -> None:
        for key, value in params.items():
            setattr(self, key, value)
        self.speaker = Speaker(params)
        self.listener = Listener(params)
        self.reward = torch.Tensor(self.batchSize)
        self.totalReward = None
        self.rlNegReward = -10*self.rlScale


    def forward(self, batch):
        self.batch = batch
        msg = self.speaker.get_message(batch)
        msg = msg.detach()
        self.pred, self.predDistr = self.listener.get_concepts(msg)
        return self.pred.detach()

    def performBackward(self, optimiser):
        self.reward.fill_(self.rlNegReward)
        # all the three attributes needs to be predicted correctly
        first_match = self.pred[:,0] == self.batch[:,0]
        second_match = self.pred[:,1] == self.batch[:,1]
        third_match = self.pred[:,2] == self.batch[:,2]
        
        self.reward[first_match & second_match & third_match] = self.rlScale

        # reinforce all the actions for speaker and listener bot
        self.speaker.reinforce(self.reward)
        self.listener.reinforce(self.reward)

        # optimise the speaker and listener
        optimiser.zero_grad()
        self.speaker.backward()
        self.listener.backward()

        # clip the gradients
        for p in self.speaker.parameters():
            p.grad.data.clamp_(-5, 5)
        
        for p in self.listener.parameters():
            p.grad.data.clamp_(-5, 5)
        

        # cumulative reward
        batchReward = torch.mean(self.reward)/self.rlScale
        if self.totalReward is None: self.totalReward = batchReward
        self.totalReward = self.totalReward*0.95 + batchReward*0.05