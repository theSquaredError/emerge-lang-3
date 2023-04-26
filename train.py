import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

def get_reward(concepts, pred_concepts, chosen_order):
    '''
        Returns the reward for the speaker and listener
        Current case:
        concept shape  = (batch_size, 1)
    '''
    pos_r = 10
    neg_r = -10
    rewards = []
    batch_size = concepts.shape[0]
    for i in range(batch_size):
        r = 0
        if concepts[i][0] == pred_concepts[i][0] and concepts[i][1] == pred_concepts[i][1] and concepts[i][2] == pred_concepts[i][2]:
            rewards.append(pos_r)
        else:
            rewards.append(neg_r)
    return rewards
   


def vpg(rewards, log_probs, optimiser):
    
    gamma = 1
    rewards = torch.tensor(rewards)
    # print(f'rewards = {rewards}')
    for index,log_prob in enumerate(log_probs):
        log_probs[index] = log_prob*rewards
    
    # performing backward
    optimiser.zero_grad()
    obj = sum([ii.sum() for ii in log_probs])
    obj.backward(retain_graph=True)
    # rewards = torch.sum(rewards,dim = 0)
    # log_probs_i = torch.stack(log_probs_i)
    # log_probs_i = torch.stack(l_log_probs_i)
    # s_log_prob = s_log_probs_i*rewards.sum()
    # rewards = torch.cumsum(rewards, dim = 0)
    # log_probs_i = torch.cumsum(log_probs_i, dim = 0)
    # log_prob = log_probs*rewards
    # log_prob = torch.cumsum(log_prob, dim = 0)
    # r = np.full(len(rewards), gamma) ** np.arange(len(rewards)) * np.array(rewards)
    # r = r[::-1].cumsum()[::-1]
    # discounted_rewards = torch.tensor(r - r.mean())
    # discounted_rewards = torch.tensor(rewards)
    # selected_log_probs = discounted_rewards*log_probs
    # loss = -selected_log_probs.sum()
    # + 0.01*entropy.sum()
    # loss = -(rewards.detach()*100*log_probs).sum() 
    # loss.backward()
    return obj



def data_gen(env, opts, world):
    batch_size = opts.batch_size
    data = []
    segments = []
    sectors = []
    colors = []
    for i in range(batch_size):
        obv = env.reset()
        comm_order = opts.comm_order
        octant, segment, quadrant, color = world.get_concepts(env.target_index, env.source_index)
        
        sec_e = world.get_concept_enc(octant, 'sector', opts) 
        seg_e = world.get_concept_enc(segment, 'segment', opts)
        col_e = world.get_concept_enc(color, 'color', opts)
        
        # get input
        input = []
        # for i,c in enumerate(comm_order):
        #     if c == 'sector':
        #         input.append(torch.Tensor(sec_e))
        #     elif c == 'segment':
        #         input.append(torch.Tensor(seg_e))
        #     elif c == 'color':
        #         input.append(torch.Tensor(col_e))
        segments.append(seg_e)
        sectors.append(sec_e)
        colors.append(col_e)
        input = seg_e + sec_e + col_e
        input = torch.Tensor(input)
        # input = torch.stack(input)
        data.append(input)
    segments = torch.Tensor(segments)
    sectors = torch.Tensor(sectors)
    colors = torch.Tensor(colors)
    data = torch.stack(data)
    # getting data in the shape (3, batch_size, attribute value size)
    # print(f'data shape {data.shape}') #(10,12)
    # data = data.permute(1,0,2)
    # data shape = (comm_order, batch_size, n_concepts)
    return data

def train(env, speaker, listener, opts, world, n_episodes = 1, epochs = 100000):
    comm_order = opts.comm_order
    dataLoader = opts.dataLoader
    optimizer1 = torch.optim.Adam(speaker.parameters(), lr = opts.lr)
    optimizer2 = torch.optim.Adam(listener.parameters(), lr = opts.lr)
    
    for e in range(epochs):
        s_data = dataLoader.getBatch(opts.batch_size)
        # run this data for speaker and get listener's data
        msg, s_log_probs = speaker(s_data, comm_order)

        pred_concepts, l_log_probs, l_entropy = listener(msg, comm_order)
        rewards =  get_reward(s_data, pred_concepts,  opts)
        # print(f'rewards {rewards}')
        # train the speaker and listener
        s_loss = vpg(rewards, s_log_probs, optimiser=optimizer1)
        l_loss = vpg(rewards, l_log_probs, optimiser=optimizer2)
        optimizer1.step()
        optimizer2.step()

        if e%10 == 0:
            print('='*80)
            print(f'Epoch {e}: s_loss={s_loss}, l_loss={l_loss}')
            cnt = Counter()
            for r in rewards:
                cnt[r]+=1
            print(f'cnt {cnt}')
            print(f'log_probs {s_log_probs[:10]}')
            # print(f'rewards mean {torch.sum(torch.tensor(rewards, dtype=torch.float32), dim = 0).mean()}')
            print(f'rewards mean {torch.mean(torch.tensor(rewards, dtype=torch.float32), dim = 0)}')

