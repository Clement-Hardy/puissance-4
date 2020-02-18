import numpy as np
import torch


class ReplayBuffer(object):
    
    def __init__(self, max_data, batch_size):
        
        self.max_data = max_data      # memory size of the buffer
        self.data = [] 
        self.pos = 0                  # the current position in the list containing the data
        self.batch_size = batch_size  # the minibatch size

    def append_data(self, state, action, reward, next_state, finish):
        """
            Store a new experience in the buffer.
            Params
            ------
            state: S_t
            action: A_t
            reward: R_t+1
            next_state: S_t+1
            finish: 1 if the action lead to the end of the episode, 0 otherwise
        """        
        if len(self.data) < self.max_data:
            # if the list isn't already full we simply push the result 
            # in the end of the list
            self.data.append(tuple((state, action, reward,
                                    next_state, finish)))
        else:
            # otherwise we replace the data at the position "pos" of the list
            self.data[self.pos] = tuple((state, action, reward, next_state, finish))
        self.pos += 1
        self.pos %= self.max_data  # when we arrive at the end of the list we restart at the beginning of the list
        # like that the old result is deleted from the list.
        
    def sample(self):
        """
            Sample a minibatch from Replay Buffer.
            Returns
            -------
            states: float tensor
                Batch of observations
            actions: float tensor
                Batch of actions executed given states
            rewards: float tensor
                Rewards received as results of executing actions
            next_states: np.array
                Next set of observations seen after executing actions
            finish: float tensor
                finish[i] = 1 if executing actions[i] resulted in the end of an 
                episode and 0 otherwise.
            """
        
        # the experiences of the minibatch are choosed randomly (the minibatch has the size batch_size)
        indices = np.random.randint(0, len(self.data), self.batch_size)
        states, actions, rewards, next_states, finishs = [], [], [], [], []
        
        # we add the experience in the minibatch
        for i in indices:
            states.append(self.data[i][0])
            actions.append(self.data[i][1])
            rewards.append(self.data[i][2])
            next_states.append(self.data[i][3])
            finishs.append(self.data[i][4])
        
        # converting numpy arrays to float tensors (pytorch can't work with numpy array)
        return states, torch.FloatTensor(actions), torch.FloatTensor(rewards), \
            next_states,  torch.FloatTensor(finishs)
