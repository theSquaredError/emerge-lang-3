import torch.nn as nn
import numpy as np
import torch

def choose_concept(vocab, world, opts,  policy, hidden_l):
    vocab_encoding = world.get_vocab_tensor(vocab)
    output_l, hidden_l = policy.lnet(vocab_encoding, hidden_l)
    

    concept_i = np.random.choice(np.arange(opts.n_concepts), p = output_l.detach().numpy())
    concept = world.all_concepts[concept_i]
    action_l = concept_i
    return concept, output_l, hidden_l, action_l
    
def choose_vocab(concept, world, opts, policy, hidden_s):
    concept_encoding = world.get_concept_tensor(concept)
    output_s, hidden_s = policy.snet(concept_encoding, hidden_s)
    vocab_i = np.random.choice(np.arange(opts.n_vocab), p = output_s.detach().numpy())
    action_s = vocab_i
    vocab = world.vocabularies[vocab_i]
    return vocab, output_s, hidden_s, action_s


def run_episode(policy, obv, world, opts, env):
    reward = []
    # pass this obv to speaker
    # get the order for concept utterance
    order_probs = policy.order(obv)
    # get the concept
    chosen_order = np.random.choice(np.arange(opts.order_vec_size), p = order_probs.detach().numpy())
    chosen_order = opts.order_vec[chosen_order]
    # get the real concepts from obv
    octant, segment, quadrant, color = world.get_concepts(env.target_index, env.source_index)

    # perform the communication between speaker and listener
    for c in chosen_order:
        
        if c == 'sector':
            vocab = choose_vocab(octant, policy)

            # listener
            pred_concept = choose_concept(vocab, policy)

            if pred_concept == octant:
                reward.append(10)
            else:
                reward.append(-10)
        
        if c == 'segment':
            vocab = choose_vocab(segment, policy)

            # listener
            pred_concept = choose_concept(vocab, policy)

            if pred_concept == segment:
                reward.append(10)
            else:
                reward.append(-10)

        if c == 'color':
            vocab = choose_vocab(color, policy)

            # listener
            pred_concept = choose_concept(vocab, policy)

            if pred_concept == color:
                reward.append(10)
            else:
                reward.append(-10)
                
    return reward

def train_centralised_policy(policy, env, epochs = 1000):
    optim = torch.optim.Adam(policy.parameters(), lr = 0.001)
    for _ in range(epochs):
        # reset the environment
        obv = env.reset()
        
    pass
